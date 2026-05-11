# qq adapter readable mentions plan

## Summary

- Goal: Execute the approved Option 2 behavior by making adapter-normalized
  mention text readable before `/chat`, so the brain and RAG see names instead
  of CQ or platform mention syntax.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, `test-style-and-execution`, and `character-test`
  for any debug-channel or service-level live validation.
- Overall cutover strategy: bigbang for adapter-normalized inbound message
  text; compatible for the existing `/chat` schema.
- Highest-risk areas: accidental semantic rewriting in adapters, leaking CQ or
  Discord syntax into `body_text`, adding brain/RAG repair logic, and adapter
  display-name lookup latency.
- Acceptance criteria: QQ and Discord adapters follow the same readable
  mention protocol, `/chat` ICDs document that protocol, deterministic tests
  pass, and one real LLM validation shows the display-name input drives
  person-context retrieval.

## Context

RCA for QQ message ObjectId `6a019c76f40aef170e8a8cb8` showed the original
QQ wire message was:

```text
[CQ:at,qq=3768713357] 你怎么评价群友[CQ:at,qq=673225019]
```

The current QQ adapter stripped both CQ mentions from `body_text`, so the brain
saw only:

```text
你怎么评价群友
```

The decontextualizer and RAG then operated on incomplete semantic input. RAG
was not the failing boundary. A real LLM feasibility check showed that when the
input is adapter-normalized as readable mention text:

```text
@杏山千纱 你怎么评价群友 @蚝爹油
```

the RAG initializer selects person-context retrieval for `蚝爹油`, and the full
RAG path can resolve the user through existing display-name lookup and profile
agents. A separate real LLM check showed that a QQ-id-style fallback is not
resolved by current RAG. Therefore the adapter must preserve visible mention
targets as readable display labels where the platform can provide or resolve
them.

This plan keeps semantic judgment out of adapters. The adapter only converts
platform mention tokens into platform-neutral readable mention tokens. It must
not infer that `群友` means a particular user, must not rewrite pronouns, and
must not change non-mention authored text.

## Mandatory Skills

- `development-plan-writing`: load before editing this plan or lifecycle
  records.
- `local-llm-architecture`: load before implementation to preserve the
  adapter/brain/RAG responsibility boundary and local-LLM input-shaping rules.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK test strings
  or prompt-facing examples.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `character-test`: load before any service/debug-channel validation that
  exercises real chatbot behavior.

## Mandatory Rules

- Runtime behavior changes must stay inside `src/adapters/**`.
- Documentation updates for the `/chat` and message-envelope ICDs are allowed
  outside `src/adapters/**` because they define the adapter-to-brain contract.
- Tests may be changed or added outside `src/adapters/**` only to prove the
  adapter protocol and the real LLM feasibility gate.
- Do not change RAG, decontextualizer, cognition, dialog, persistence, service
  queueing, prompt projection, or brain-service runtime code.
- Do not add semantic replacement logic. The adapter may replace a platform
  mention token with a readable mention token only.
- Do not put CQ, Discord mention tags, raw platform IDs, or platform names into
  `message_envelope.body_text`.
- QQ mention display labels must follow one consistent adapter source policy:
  use the same nickname-first source family as QQ top-level sender
  `display_name`; use group card only when nickname-style labels are absent.
- Do not change the `ChatRequest` or `MessageEnvelope` schema in code. Use the
  existing `mentions[].display_name`, `mentions[].platform_user_id`,
  `mentions[].global_user_id`, and `raw_wire_text` fields.
- Keep `raw_wire_text` as the audit/replay field that may contain platform wire
  syntax.
- Treat live LLM checks as one-case-at-a-time tests. Inspect the trace before
  recording the gate as passed.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or
  final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.

## Must Do

- Update the adapter-readable mention protocol in the `/chat` ICD.
- Update the message-envelope ICD so visible mentions are represented as
  readable mention tokens in `body_text`, while raw platform syntax remains
  only in `raw_wire_text`.
- Add or update deterministic tests proving QQ and Discord follow the same
  readable mention protocol.
- Update QQ adapter normalization so CQ mention tokens become readable
  `@display_name` tokens in `body_text`.
- Update QQ adapter event handling to hydrate mention display names from a
  consistent nickname-first QQ source policy before `/chat`.
- Update Discord adapter normalization so Discord user, role, channel, and
  everyone/here mentions become readable platform-neutral tokens in
  `body_text`.
- Preserve typed mention records, including `display_name` when available.
- Run the focused deterministic adapter and runtime-adapter tests.
- Run one real LLM validation that uses adapter-shaped body text and confirms
  person-context retrieval targets the mentioned display name.

## Deferred

- Do not modify RAG to resolve QQ IDs.
- Do not modify the decontextualizer to interpret CQ syntax, platform IDs, or
  mention metadata.
- Do not add brain-service runtime hydration for missing mention display names.
- Do not change database schema or run a data migration.
- Do not add a compatibility mode that sends stripped mention text for old
  behavior.
- Do not redesign prompt-message projection or remove existing typed mention
  metadata from prompts.
- Do not introduce profile-backed mention resolution inside adapters beyond the
  platform display-name lookup described here.

## Cutover Policy

Overall strategy: bigbang for adapter-normalized `body_text`, compatible for
the existing `/chat` schema.

| Area | Policy | Instruction |
| --- | --- | --- |
| QQ inbound mention text | bigbang | Replace stripped CQ mention behavior with readable mention tokens. Do not preserve a stripped-text path. |
| Discord inbound mention text | bigbang | Replace stripped Discord mention behavior with readable mention tokens. Do not preserve a stripped-text path. |
| `/chat` schema | compatible | Keep existing fields and Pydantic contracts. Clarify semantics in ICD docs only. |
| Mention lookup failure | compatible | Use platform-neutral occurrence labels such as `@mentioned-user-1`; keep the platform id only in typed metadata. |
| RAG and cognition | bigbang exclusion | No runtime code changes are allowed in these layers for this plan. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- For bigbang adapter behavior, rewrite tests to the new contract instead of
  preserving old stripped-mention assertions.
- The only allowed mention lookup failure representation in `body_text` is a
  platform-neutral occurrence label.
- Any change to this cutover policy requires user approval before
  implementation.

## Agent Autonomy Boundaries

- The target ownership boundary is `src/adapters/**`.
- The agent may choose local helper mechanics only when they preserve the exact
  protocol in this plan and stay in `src/adapters/**`.
- The agent must search existing adapter helpers before adding a helper.
- The agent must not introduce new service fields, new prompt inputs, new
  adapter-to-brain schema fields, feature flags, alternate brain paths, or
  background jobs.
- If a display-name source is unavailable, the agent must degrade to the
  platform-neutral occurrence label and log the lookup miss at adapter debug or
  warning level according to existing adapter style.
- If implementing a fix requires changing non-adapter runtime code, stop and
  report the blocker instead of expanding scope.
- Documentation and test edits must stay limited to the protocol and
  verification files named in this plan.

## Target State

All adapters that parse platform mention grammar send `/chat` requests with
these invariants. Pass-through adapters, such as the debug adapter, satisfy the
same protocol by not introducing platform mention syntax or synthetic mention
metadata.

- `message_envelope.body_text` contains authored text plus readable visible
  mention tokens.
- `body_text` contains no CQ codes, Discord `<@...>` tags, raw platform IDs,
  raw platform names, native reply markers, channel tags, role tags, or custom
  emoji wire syntax.
- `message_envelope.raw_wire_text` preserves the original or closest replayable
  platform wire text.
- `message_envelope.mentions[]` carries platform identity, resolved global
  identity when known, display label when known, entity kind, and raw mention
  text.
- QQ example:

```json
{
  "body_text": "@杏山千纱 你怎么评价群友 @蚝爹油",
  "raw_wire_text": "[CQ:at,qq=3768713357] 你怎么评价群友[CQ:at,qq=673225019]",
  "mentions": [
    {
      "platform_user_id": "3768713357",
      "global_user_id": "<character_global_user_id>",
      "display_name": "杏山千纱",
      "entity_kind": "bot",
      "raw_text": "[CQ:at,qq=3768713357]"
    },
    {
      "platform_user_id": "673225019",
      "display_name": "蚝爹油",
      "entity_kind": "user",
      "raw_text": "[CQ:at,qq=673225019]"
    }
  ]
}
```

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Ownership | Adapters own mention-token readability. | Adapters are the only layer that understands platform wire grammar. |
| Schema | Reuse existing `/chat` and envelope fields. | The needed display-name slot already exists; schema churn would widen blast radius. |
| Semantic boundary | Replace only platform mention tokens. | The user explicitly rejected semantic adapter rewrites. |
| Unknown mentions | Emit `@mentioned-user-N`, `@mentioned-role-N`, `#mentioned-channel-N`, or `@mentioned-entity-N`. | This avoids leaking platform IDs or platform-specific labels into LLM-visible text. |
| QQ display-name source priority | Use bot name for bot mentions. For human QQ mentions, use nickname-style labels first from segment data, then adapter cache, then bounded NapCat/OneBot lookup; use group card only when nickname-style labels are absent; then occurrence label. | This stays consistent with the current QQ sender `display_name` source and reduces mismatch against DB profiles created from adapter display names. |
| Discord display-name source priority | Use Discord SDK objects on the message event, then occurrence label. | The SDK already exposes mentioned user, role, and channel labels without extra API calls. |
| RAG behavior | Leave RAG unchanged. | Existing RAG resolves display names; it does not reliably resolve QQ IDs. |

## Protocol Contract

Adapters must format visible mention tokens as:

| Entity kind | Body token when display name is available | Body token when display name is unavailable |
| --- | --- | --- |
| `bot` | `@<display_name>` | `@mentioned-user-N` |
| `user` | `@<display_name>` | `@mentioned-user-N` |
| `platform_role` | `@<display_name>` | `@mentioned-role-N` |
| `channel` | `#<display_name>` | `#mentioned-channel-N` |
| `everyone` | `@everyone`, `@here`, or `@all` | same as raw broadcast label without platform wrapper |
| `unknown` | `@<display_name>` | `@mentioned-entity-N` |

Display names are labels only. Adapters must trim surrounding whitespace and
collapse internal whitespace in labels. Adapters must not translate,
summarize, infer aliases, or replace ordinary authored words.

## LLM Call And Context Budget

This plan adds no LLM calls and changes no prompt, RAG, cognition, evaluator,
dialog, consolidation, or background LLM stage.

Expected effect on existing LLM context:

- Before: the LLM-visible `body_text` for the RCA message was
  `你怎么评价群友`.
- After: the LLM-visible `body_text` is
  `@杏山千纱 你怎么评价群友 @蚝爹油`.
- Context growth is bounded by the number of visible mention tokens in the
  original message. Normal chat messages have few mentions.
- Prompt-message context already projects typed mention metadata. This plan
  does not add new projected fields.

Adapter-side QQ lookup budget:

- Use no lookup when segment data or cache provides a nickname-style label.
- Lookup at most each unique mentioned QQ id once per inbound event.
- Bound each platform lookup to a short adapter-local timeout of 1 second.
- On timeout or API failure, use the occurrence label and continue.
- If a NapCat/OneBot response contains both `nickname` and `card`, use
  `nickname`. Use `card` only when `nickname` is empty.

## Change Surface

### Modify

- `src/adapters/envelope_common.py`: add or reuse adapter-boundary helper logic
  for formatting readable mention body tokens and sanitizing display labels.
- `src/adapters/napcat_qq_adapter.py`: hydrate QQ mention display names,
  cache labels, replace CQ mention tokens with readable body tokens, and fill
  mention `display_name` when known.
- `src/adapters/discord_adapter.py`: pass Discord SDK mention display labels
  into the normalizer, replace Discord mention tags with readable body tokens,
  and fill mention `display_name` when known.
- `tests/test_adapter_envelope_normalizers.py`: update QQ and Discord
  normalizer expectations to the readable mention protocol.
- `tests/test_runtime_adapter_registration.py`: update QQ runtime adapter tests
  and add focused coverage for segment-data labels, lookup labels, and lookup
  failure occurrence labels.
- `src/kazusa_ai_chatbot/brain_service/README.md`: update `/chat` adapter
  responsibilities and `message_envelope` semantics.
- `src/kazusa_ai_chatbot/message_envelope/README.md`: update `body_text`,
  `raw_wire_text`, and `mentions` rules for readable mention tokens.

### Create

- `tests/test_adapter_readable_mentions_live_llm.py`: one live LLM contract
  test or diagnostic that feeds adapter-shaped readable mention text through
  decontextualizer and RAG initializer, writes a trace, and asserts
  person-context retrieval targets the display name `蚝爹油`.

### Keep

- Keep `src/kazusa_ai_chatbot/brain_service/contracts.py` unchanged.
- Keep `src/kazusa_ai_chatbot/message_envelope/types.py` unchanged.
- Keep `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`
  unchanged.
- Keep `src/kazusa_ai_chatbot/rag/**` unchanged.
- Keep `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  unchanged.
- Keep `src/kazusa_ai_chatbot/service.py` and
  `src/kazusa_ai_chatbot/brain_service/intake.py` unchanged.
- Keep `src/adapters/debug_adapter.py` unchanged unless a doc-test proves a
  direct contradiction. The debug adapter already sends user-entered text as
  plain text and has no platform mention grammar to normalize.

## Implementation Order

1. Update deterministic adapter tests first.
   - File: `tests/test_adapter_envelope_normalizers.py`.
   - Add QQ normalizer expectations for display-name replacement and occurrence
     label replacement.
   - Add Discord normalizer expectations for user, role, channel, and
     everyone/here readable tokens.
   - Run:
     `venv\Scripts\python -m pytest tests/test_adapter_envelope_normalizers.py -q`
   - Expected before implementation: tests fail on old stripped mention body
     text.

2. Update QQ runtime adapter tests.
   - File: `tests/test_runtime_adapter_registration.py`.
   - Update old stripped-body assertions.
   - Add or extend tests for segment-list display names, API lookup labels,
     cache reuse, lookup failure occurrence labels, and conflicting
     `nickname`/`card` values.
   - The conflicting `nickname`/`card` case must assert that the readable
     mention token and `mentions[].display_name` use `nickname`.
   - Run the touched tests by node id.
   - Expected before implementation: tests fail because CQ mentions are still
     stripped or no display label is hydrated.

3. Implement shared adapter mention-label helper.
   - File: `src/adapters/envelope_common.py`.
   - Add a small helper that formats readable mention body tokens according to
     `Protocol Contract`.
   - Add no imports from brain, RAG, service, nodes, or database modules beyond
     existing adapter-boundary imports.

4. Implement QQ adapter changes.
   - File: `src/adapters/napcat_qq_adapter.py`.
   - Compute group/private channel context before mention hydration.
   - Preserve CQ wire text in `raw_wire_text`.
   - Use `self.bot_name` for bot-id mentions when available.
   - Add an adapter-local cache keyed by `(channel_id, platform_user_id)`.
   - Store the cache value only from the selected QQ source policy, not from
     arbitrary first-seen labels.
   - Build human-user labels from segment data with `nickname` first, then
     `name`, then `card` only when nickname-style labels are absent.
   - For unresolved group mentions, call NapCat/OneBot
     `get_group_member_info` with `group_id` and `user_id`.
   - For unresolved private mentions, call NapCat/OneBot `get_stranger_info`
     with `user_id`.
   - For lookup responses, choose `nickname` before `card`.
   - Bound each lookup to 1 second and continue on timeout or non-ok response.
   - Replace each CQ mention in `body_text` with the readable token while
     leaving non-mention CQ codes stripped as before.

5. Implement Discord adapter changes.
   - File: `src/adapters/discord_adapter.py`.
   - Build display-name maps from `message.mentions`,
     `message.role_mentions`, and `message.channel_mentions`.
   - Replace user, role, channel, and everyone/here tags with readable tokens.
   - Keep custom emoji tags stripped as transport syntax.

6. Update ICD documentation.
   - Files:
     `src/kazusa_ai_chatbot/brain_service/README.md` and
     `src/kazusa_ai_chatbot/message_envelope/README.md`.
   - Document that visible mentions are authored mention content and must be
     represented as readable platform-neutral tokens in `body_text`.
   - Document that raw platform mention syntax belongs only in
     `raw_wire_text`.
   - Document that adapter lookup failure uses occurrence labels, not raw
     platform IDs.

7. Add live LLM validation.
   - File: `tests/test_adapter_readable_mentions_live_llm.py`.
   - Build the state from adapter-shaped text:
     `@杏山千纱 你怎么评价群友 @蚝爹油`.
   - Run decontextualizer and RAG initializer, or the smallest existing RAG
     path that proves the slot becomes person-context for `蚝爹油`.
   - Write a trace under `test_artifacts/llm_traces/`.
   - Mark with `pytest.mark.live_llm`.

8. Run verification gates and record evidence.
   - Execute every command listed in `Verification`.
   - Inspect the live LLM trace before marking that gate passed.
   - Record results in `Execution Evidence`.

9. Run independent code review.
   - Follow the `Independent Code Review` section.
   - Fix only findings inside this plan's change surface.
   - Rerun affected verification commands and update evidence.

## Progress Checklist

- [x] Stage 1 - deterministic protocol tests updated.
  - Covers: implementation steps 1 and 2.
  - Verify: focused adapter tests fail before implementation for the old
    stripped-mention behavior.
  - Evidence: record failing test names and failure summaries in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-12` after evidence is recorded.

- [x] Stage 2 - adapter runtime implementation complete.
  - Covers: implementation steps 3, 4, and 5.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_adapter_envelope_normalizers.py -q`
    and touched `tests/test_runtime_adapter_registration.py` tests pass.
  - Evidence: record changed adapter files and command output summaries.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex / 2026-05-12` after evidence is recorded.

- [x] Stage 3 - ICD documentation updated.
  - Covers: implementation step 6.
  - Verify: manual diff review confirms `/chat` and message-envelope docs
    match the protocol and do not describe old stripped-mention behavior.
  - Evidence: record doc paths and summary.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex / 2026-05-12` after evidence is recorded.

- [x] Stage 4 - live LLM validation complete.
  - Covers: implementation step 7.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_adapter_readable_mentions_live_llm.py -q -m live_llm`
    runs one case and writes a trace.
  - Evidence: record trace path, decontextualized input, RAG initializer slot,
    and whether person-context targeted `蚝爹油`.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex / 2026-05-12` after trace inspection and evidence are recorded.

- [x] Stage 5 - final verification and independent code review complete.
  - Covers: implementation steps 8 and 9.
  - Verify: every `Verification` command passes or has its documented allowed
    skip condition, and independent review approves completion.
  - Evidence: record final test output summaries, review findings, fixes,
    rerun commands, and residual risks.
  - Handoff: plan may be marked completed only after this stage is signed off.
  - Sign-off: `Codex / 2026-05-12` after evidence and review approval are recorded.

## Verification

### Deterministic Tests

- `venv\Scripts\python -m pytest tests/test_adapter_envelope_normalizers.py -q`
  - Expected: passes.
- `venv\Scripts\python -m pytest tests/test_runtime_adapter_registration.py -q`
  - Expected: passes.

### Live LLM Test

- `venv\Scripts\python -m pytest tests/test_adapter_readable_mentions_live_llm.py -q -m live_llm`
  - Expected: runs one live case when the configured LLM endpoint is available.
  - Allowed skip: the test may skip only when the configured live LLM endpoint
    is unavailable under the existing live-test skip pattern.
  - Required trace evidence: the trace shows the LLM-facing body text contains
    `@蚝爹油`, the decontextualizer does not drop that display label, and the
    RAG initializer emits a person-context need for `蚝爹油`.

### Static Greps

- `rg "\[CQ:|<@!?|<@&|<#" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag tests\test_adapter_readable_mentions_live_llm.py`
  - Expected: zero matches. New live LLM payloads and brain/RAG prompt code
    must not contain platform wire syntax.
  - Allowed nonzero exit: `rg` exit code 1 when no matches are found.
- `rg "get_group_member_info|get_stranger_info" src\kazusa_ai_chatbot`
  - Expected: zero matches. Platform lookup calls must stay in
    `src/adapters/napcat_qq_adapter.py`.

### Runtime Boundary Review

- `git diff --name-only`
  - Expected runtime-code paths: only files under `src/adapters/**`.
  - Expected non-runtime paths: the two ICD docs and tests listed in
    `Change Surface`.
  - Forbidden paths: `src/kazusa_ai_chatbot/rag/**`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`,
    `src/kazusa_ai_chatbot/service.py`,
    `src/kazusa_ai_chatbot/brain_service/intake.py`, and
    `src/kazusa_ai_chatbot/brain_service/contracts.py`.

## Independent Plan Review

Review mode: self-review from a fresh approval posture. No separate reviewer
was available in this session.

Inputs reviewed:

- This plan.
- `development_plans/README.md`.
- `src/adapters/napcat_qq_adapter.py`.
- `src/adapters/discord_adapter.py`.
- `src/adapters/debug_adapter.py`.
- `src/adapters/envelope_common.py`.
- `src/kazusa_ai_chatbot/brain_service/README.md`.
- `src/kazusa_ai_chatbot/message_envelope/README.md`.
- `tests/test_adapter_envelope_normalizers.py`.
- `tests/test_runtime_adapter_registration.py`.

Review findings:

- Blockers: none.
- Non-blocking finding fixed: clarified that pass-through adapters satisfy the
  shared protocol by not introducing platform mention syntax, rather than by
  parsing arbitrary user-authored plain text.
- Non-blocking finding fixed: plan and registry status changed from `draft` to
  `approved`.
- Residual risk: implementation still needs real NapCat API-shape verification
  through the planned adapter tests, because display-name field availability
  can vary by NapCat/OneBot event shape.

Approval status: approved for implementation under the change surface and
verification gates in this plan.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed adapter, test, and
  documentation artifact.
- Code quality and design weaknesses, including adapter ownership boundaries,
  hidden semantic replacement, raw platform syntax leakage, platform-id leakage
  into `body_text`, unbounded NapCat API calls, brittle fixtures, and avoidable
  blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including trace evidence, focused tests,
  static grep expectations, and lifecycle records.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- QQ `body_text` for the RCA shape includes readable mention tokens for the bot
  and mentioned user.
- QQ mention labels use the nickname-first source policy; when `nickname` and
  `card` conflict, `body_text` and `mentions[].display_name` use `nickname`.
- Discord `body_text` includes readable mention tokens for visible user, role,
  channel, and everyone/here mentions.
- No adapter sends CQ or Discord wire mention syntax in `body_text`.
- No adapter sends raw platform IDs or platform names in `body_text` as a
  mention lookup failure representation.
- Typed mention metadata still carries platform ids and display names where
  available.
- `/chat` and message-envelope ICDs define the shared adapter protocol.
- RAG, decontextualizer, service runtime, and brain-service runtime contracts
  are unchanged.
- Deterministic adapter tests and runtime adapter tests pass.
- The live LLM validation trace confirms display-name mention text causes
  person-context retrieval for `蚝爹油`.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Adapter starts doing semantic rewrites | Restrict replacement to platform mention tokens only | Deterministic tests compare surrounding authored text exactly |
| NapCat lookup adds response latency | Use segment labels and cache first; bound lookup to 1 second per unique id | Runtime adapter tests and code review inspect timeout path |
| QQ card differs from DB profile display name | Use nickname-first policy to match current QQ sender display-name source; use card only when nickname is absent | Runtime adapter conflict test asserts nickname wins over card |
| Display label unavailable | Emit `@mentioned-user-N` and keep id in typed metadata | Lookup failure test asserts no raw platform id in `body_text` |
| Discord and QQ diverge | Use the same protocol table and shared adapter helper | QQ and Discord normalizer tests assert the same token rules |
| ICD says one thing and code does another | Update ICD after tests and implementation, then perform manual diff review | Stage 3 and independent review gates |
| Live LLM result is flaky | Run one focused case and inspect trace, treating it as validation evidence rather than broad regression coverage | Live LLM gate records raw trace path and parsed outputs |

## Execution Evidence

- Plan approval review: completed in self-review mode. Blockers: none.
  Fixes applied: clarified pass-through adapter protocol scope; changed plan
  and registry status to approved. Residual risk: NapCat display-name field
  shape must be verified during implementation.
- Plan amendment after user source-consistency clarification: QQ display-name
  source policy is now nickname-first to match existing sender display-name
  behavior; group card is a fallback only when nickname-style labels are
  absent. Added required conflicting `nickname`/`card` test coverage.
- Stage 1 evidence: updated `tests/test_adapter_envelope_normalizers.py` and
  `tests/test_runtime_adapter_registration.py`. Before implementation,
  `venv\Scripts\python -m pytest tests\test_adapter_envelope_normalizers.py -q`
  failed three readable-mention assertions and passed the plain group-message
  broadcast test. Before implementation,
  `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py::test_napcat_handle_event_sends_readable_bot_mention_and_typed_envelope tests\test_runtime_adapter_registration.py::test_napcat_handle_event_hydrates_human_mention_nickname_and_cache tests\test_runtime_adapter_registration.py::test_napcat_handle_event_uses_occurrence_label_when_lookup_fails -q`
  failed all three expected readable-mention assertions because QQ mentions
  were still stripped.
- Stage 2 evidence: implemented readable mention helpers in
  `src/adapters/envelope_common.py`, QQ CQ mention replacement and bounded
  NapCat display-name hydration/cache in `src/adapters/napcat_qq_adapter.py`,
  and Discord user/role/channel/everyone readable mention replacement in
  `src/adapters/discord_adapter.py`. Syntax check
  `venv\Scripts\python -m py_compile src\adapters\envelope_common.py src\adapters\napcat_qq_adapter.py src\adapters\discord_adapter.py tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py`
  passed. Deterministic normalizer command
  `venv\Scripts\python -m pytest tests\test_adapter_envelope_normalizers.py -q`
  passed 4 tests. Focused runtime command
  `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py::test_napcat_handle_event_sends_readable_bot_mention_and_typed_envelope tests\test_runtime_adapter_registration.py::test_napcat_handle_event_uses_segment_nickname_without_lookup tests\test_runtime_adapter_registration.py::test_napcat_handle_event_hydrates_human_mention_nickname_and_cache tests\test_runtime_adapter_registration.py::test_napcat_handle_event_uses_occurrence_label_when_lookup_fails -q`
  passed 4 tests.
- Stage 3 evidence: updated `src/kazusa_ai_chatbot/brain_service/README.md`
  and `src/kazusa_ai_chatbot/message_envelope/README.md`. Manual diff review
  confirmed `/chat` adapter responsibilities and message-envelope semantics now
  require readable mention tokens in `body_text`, raw platform syntax only in
  `raw_wire_text`, occurrence labels for lookup misses, and no adapter semantic
  rewrites.
- Stage 4 evidence: added
  `tests/test_adapter_readable_mentions_live_llm.py`. Command
  `venv\Scripts\python -m pytest tests\test_adapter_readable_mentions_live_llm.py -q -m live_llm -s`
  passed one live case. Trace inspected at
  `test_artifacts/llm_traces/adapter_readable_mentions_live_llm__qq_named_user_person_context.json`.
  The trace showed `adapter_shaped_body_text` and
  `decontextualized_input` both as `@杏山千纱 你怎么评价群友 @蚝爹油`; RAG
  initializer emitted
  `Person-context: retrieve profile/impression for display name 蚝爹油`.
  Judgment: person-context targeted `蚝爹油` successfully.
- Stage 5 final verification:
  `venv\Scripts\python -m py_compile src\adapters\envelope_common.py src\adapters\napcat_qq_adapter.py src\adapters\discord_adapter.py tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py tests\test_adapter_readable_mentions_live_llm.py`
  passed.
  `venv\Scripts\python -m pytest tests\test_adapter_envelope_normalizers.py -q`
  passed 4 tests.
  `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
  passed 30 tests.
  `venv\Scripts\python -m pytest tests\test_adapter_readable_mentions_live_llm.py::test_live_adapter_readable_mentions_drive_person_context -q -s -m live_llm`
  passed one individually run live case. New trace inspected at
  `test_artifacts/llm_traces/adapter_readable_mentions_live_llm__qq_named_user_person_context__20260511T202847438836Z.json`;
  it preserved `@蚝爹油` through decontextualization and emitted
  `Person-context: retrieve profile/impression for display name 蚝爹油`.
  Static grep
  `rg "\[CQ:|<@!?|<@&|<#" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag tests\test_adapter_readable_mentions_live_llm.py`
  returned no matches with expected `rg` exit code 1. Static grep
  `rg "get_group_member_info|get_stranger_info" src\kazusa_ai_chatbot`
  returned no matches with expected `rg` exit code 1. Syntax checks passed for
  changed Python files, including a final
  `venv\Scripts\python -m py_compile src\adapters\napcat_qq_adapter.py` after
  the review wording fix.
- Independent code review: two independent reviewer agents inspected the
  completed diff. Runtime/boundary review initially returned approve-with-fixes
  because positive QQ mention label cache retention was unbounded and repeated
  unresolved mentions of the same QQ id in one message could repeat bounded
  lookup cost. Tests/docs/lifecycle review initially returned approve-with-fixes
  because timeout fallback lacked deterministic coverage and local adapter
  docstrings still described strip-only behavior. Fixes applied:
  `NapCatWSAdapter` now uses bounded LRU mention-label cache retention,
  suppresses duplicate unresolved lookup attempts per inbound message, imports
  `WebSocketException` directly so timeout/error paths are catchable in this
  environment, updates QQ/Discord normalizer docstrings, and adds deterministic
  tests for cache eviction plus duplicate timeout fallback with no raw QQ id
  leakage. Both reviewers re-reviewed the follow-up diff and approved. Residual
  risk: messages with many distinct uncached QQ mentions may still spend one
  bounded lookup timeout per distinct id, and real NapCat event display-name
  fields can vary by deployment. Approval status: completed.
