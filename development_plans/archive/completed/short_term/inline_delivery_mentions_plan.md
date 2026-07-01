# inline delivery mentions plan
## Summary

- Goal: Replace prefix-only `delivery_mentions` with inline adapter-rendered
  mentions driven by dialog-authored `@display_name` text.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `debug-llm`,
  `py-style`, `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for `delivery_mentions` semantics, dialog
  prompt wording, adapter mention rendering, tests, and ICD wording.
- Highest-risk areas: adapter identity leakage, dialog-owned delivery
  mechanics, adapter database lookup, fenced-code corruption, and conflict
  with `dialog_message_sequence_delivery_plan.md`.
- Acceptance criteria: dialog may author one or more `@display_name` tags;
  brain exposes only renderable user mention candidates; adapters replace exact
  inline tokens; prefix behavior and `mention_target_user` delivery semantics
  are removed.

## Context

Kazusa currently has two separate mention concepts:

- Inbound adapter normalization preserves platform mentions as readable
  `@display_name` body text plus typed `message_envelope.mentions`.
- Outbound delivery uses `delivery_mentions` as a prefix-only request generated
  from `mention_target_user`.

The outbound behavior is not aligned with the desired ownership model. The
dialog model should decide whether visible text tags a person by writing the
tag sign in final text. The adapter should only translate a known, exact
platform-neutral tag token into the native platform mention mechanism.

Current implementation facts:

- `ChatResponse.delivery_mentions` is an untyped `list[dict]`.
- `dialog_agent.py` currently forbids visible `@` in final dialog and asks for
  `mention_target_user`.
- `service.py` converts `mention_target_user=true` into one prefix mention for
  the current user.
- Discord and NapCat QQ adapters require `placement="prefix"` and render only
  one prefix user mention.
- Dispatcher, self-cognition, and background result-ready delivery pass the
  same `delivery_mentions` list through adapter-owned send boundaries.

The active draft plan
`development_plans/active/short_term/dialog_message_sequence_delivery_plan.md`
overlaps with this plan in dialog prompt tests, Discord delivery, NapCat
delivery, Brain Service ICD wording, and mention tests. That plan currently
preserves prefix mention behavior. This plan owns the new outbound mention
semantics. The sequence plan must not be executed with its prefix-mention
contract unchanged after this plan is approved.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  signing off, or handing off this plan.
- `local-llm-architecture`: load before changing dialog prompt wording, LLM
  output contracts, response-path context, or adapter/brain ownership
  boundaries.
- `debug-llm`: load before running live LLM prompt checks or writing
  human-readable prompt quality artifacts.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests containing CJK prompt
  strings or CJK test data.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Implementation is user-approved for single-agent fallback execution as of
  this plan status change to `in_progress`.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run this plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Use single-agent fallback execution because the user explicitly requested
  execution without subagents. Record review findings in `Execution Evidence`.
- Keep the brain platform-agnostic. Dialog prompt wording may say ordinary
  online text tags such as `@display_name`; it must not mention Discord, QQ,
  OneBot, NapCat, snowflakes, CQ codes, or adapter-specific syntax.
- Keep dialog as a renderer. Dialog may decide visible wording, including
  whether the visible text contains `@display_name`, but it must not decide
  platform feasibility, native mention syntax, prefix behavior, ID lookup, or
  adapter delivery mechanics.
- Deterministic code owns mention candidate construction, ambiguity handling,
  exact token matching, platform rendering, delivery receipts, and no-op
  fallback.
- Do not give adapters database access or profile lookup responsibility.
- Do not disclose `global_user_id`, provenance fields, source refs, memory
  details, channel roster, or whole history identity lists to adapters for
  this feature.
- Prompt edits must be short and positive. Do not add long platform-specific
  negative examples or make a weak local model infer delivery mechanics.
- Only triple-backtick fenced code blocks are protected spans in this plan.
  All other text is eligible for mention replacement.

## Must Do

- Update the dialog generator prompt so visible dialog may use `@display_name`
  when the character intentionally tags the current user or a named participant
  already present in upstream dialog directives.
- Remove active dialog prompt wording that forbids `@` in final visible text.
- Remove `mention_target_user` as the outbound delivery trigger in active
  chat, self-cognition, and result-ready delivery paths.
- Keep the existing `ChatResponse.delivery_mentions` field name and type, but
  redefine each dict as a renderable inline mention candidate.
- Emit only these adapter-facing mention candidate fields:
  `entity_kind`, `display_name`, and `platform_user_id`.
- Remove outbound `placement`, `global_user_id`, and `requested_by` from
  adapter-facing mention metadata.
- Build mention candidates from bounded users already available to the current
  response path.
- Filter candidates to users whose exact `@display_name` token appears in the
  outbound text outside fenced code blocks.
- Support multiple tagged users in one outbound response. Do not collapse
  `delivery_mentions` to one candidate.
- Omit candidates whose `display_name` is ambiguous for the current response.
- Update Discord mention rendering to replace exact inline tokens with
  `<@platform_user_id>`.
- Update NapCat QQ mention rendering to replace exact inline tokens with
  OneBot `at` segments while preserving surrounding text.
- Keep debug and unsupported adapters no-op: the visible text remains
  `@display_name`.
- Update dispatcher, runtime callback, self-cognition, and background
  result-ready docs/tests so `delivery_mentions` is forwarded as inline
  render candidates, not prefix requests.
- Update focused tests, regression tests, and ICD docs that currently assert
  prefix mention behavior.
- Before approving or executing both this plan and
  `dialog_message_sequence_delivery_plan.md`, update the sequence plan so it
  delegates all `delivery_mentions` semantics to this plan and contains no
  prefix-mention contract.

## Deferred

- Do not change cognition chain stages, L2, L2d, RAG, relevance, memory
  lifecycle, consolidation, scheduler, reflection, or persistence semantics.
- Do not add new `ChatResponse` fields.
- Do not add per-message mention metadata.
- Do not add adapter database access, adapter profile lookup, channel roster
  fetches, or platform-side user search.
- Do not add fuzzy matching, alias matching, case-fold matching, nickname
  inference, partial-name matching, or fallback prefix tagging.
- Do not support role, channel, everyone, or bot mentions in this plan.
- Do not preserve `mention_target_user` as a compatibility shim for outbound
  delivery.
- Do not add feature flags, dual paths, old/new mention fallback mappers, or
  compatibility aliases.
- Do not change delivery receipt storage, assistant message persistence, or
  historical conversation rows.
- Do not implement multi-message sequence delivery from
  `dialog_message_sequence_delivery_plan.md` in this plan.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Dialog mention prompt | bigbang | Allow authored `@display_name` tags and remove `@` prohibition. |
| Dialog output schema | bigbang | Stop using `mention_target_user` as an active delivery contract. |
| `ChatResponse.delivery_mentions` field | compatible | Keep the field name and list type only. Change item semantics to inline render candidates. |
| Mention metadata item shape | bigbang | Emit only `entity_kind`, `display_name`, and `platform_user_id` for user candidates. |
| Service candidate construction | bigbang | Build bounded render candidates and filter to exact authored tokens. No prefix fallback. |
| Discord adapter rendering | bigbang | Replace exact inline `@display_name` tokens with native user mentions. |
| NapCat QQ adapter rendering | bigbang | Replace exact inline `@display_name` tokens with OneBot `at` segments. |
| Dispatcher/runtime callback contract | compatible | Keep the existing optional `delivery_mentions` field and single-message send shape. Change mention semantics only. |
| Self-cognition and result-ready sends | bigbang | Remove `mention_target_user` generation and use inline render candidates when text contains exact tags. |
| Tests and docs | bigbang | Rewrite prefix-placement assertions and ICD wording to the inline contract. |
| Message sequence plan overlap | bigbang | This plan supersedes prefix-mention instructions in `dialog_message_sequence_delivery_plan.md`; the sequence plan must be amended before both plans are approved or executed together. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- If an area is `bigbang`, rewrite legacy references instead of preserving
  fallback behavior.
- If an area is `compatible`, preserve only the explicitly listed public field
  names, endpoint shapes, or function parameters.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Normal `/chat` delivery:

```text
adapter event
  -> brain service /chat
  -> persona graph
  -> dialog_agent returns visible text that may contain @display_name
  -> service builds bounded render candidates for exact authored @display_name
  -> ChatResponse(messages, delivery_mentions=[candidate...])
  -> adapter replaces exact inline tag tokens with native platform mentions
  -> adapter sends unchanged text when no exact renderable candidate exists
```

Runtime callback, self-cognition, and result-ready delivery keep their
single-message send shape. They pass `delivery_mentions` through as inline
render candidates and rely on the same adapter rendering behavior.

Mention rendering is applied to each outbound text string that reaches an
adapter send boundary. Under the current normal `/chat` adapter behavior, that
string may be the joined text from `ChatResponse.messages`. If
`dialog_message_sequence_delivery_plan.md` later implements per-message
delivery, the same inline replacement helper must be applied independently to
each logical message string in that plan's send path. This plan does not
implement sequence delivery.

Persisted assistant `body_text` remains the brain-authored platform-neutral
text. Native rendered platform syntax is not written back into semantic
conversation text by this plan.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Mention decision owner | Dialog authors visible `@display_name` text | The character's surface wording decides whether a tag is part of the dialog. |
| Platform rendering owner | Adapter replaces exact tokens | Native mention syntax is platform delivery mechanics. |
| Candidate owner | Brain service builds candidates from bounded known users | The adapter is stateless and has no database access. |
| Adapter-facing fields | Use only `entity_kind`, `display_name`, `platform_user_id` | The adapter needs a visible token and platform id, not durable identity or provenance. |
| Matching rule | Exact `@display_name` outside fenced code blocks | Exact matching is inspectable and failure-closed. |
| Mention cardinality | Multiple user candidates per response | Dialog can tag more than one visible participant. |
| Ambiguous names | Omit ambiguous candidates | Incorrect native tags are worse than leaving plain text. |
| Unrenderable candidates | Leave authored text unchanged | Failed mention rendering must not block delivery. |
| Unsupported platforms | No-op rendering | Debug and future adapters can ignore the field safely. |
| Field compatibility | Keep `delivery_mentions` as `list[dict]` | Existing protocol surfaces can remain structurally stable while semantics cut over. |
| Sequence-plan coordination | This plan owns mention semantics | Prevents the active sequence plan from preserving prefix behavior. |

## Contracts And Data Shapes

### ChatResponse Mention Candidate

Keep the existing field:

```python
delivery_mentions: list[dict[str, Any]]
```

New item shape:

```json
{
  "entity_kind": "user",
  "display_name": "display label",
  "platform_user_id": "platform account id"
}
```

Required item rules:

- `entity_kind` must be `"user"`.
- `display_name` must be non-empty.
- Brain-owned producers must provide a non-empty `platform_user_id`.
- Adapters validate whether `platform_user_id` is renderable for the platform.
- The corresponding exact token is `@{display_name}`.
- Items with extra keys must not be generated by brain-owned producers.
- Adapters may ignore extra keys defensively if stale callers send them, but
  no compatibility behavior may depend on those keys.

### Candidate Source Policy

Candidate construction may use only users already bounded into the current
response context:

- current inbound user;
- typed inbound mentions in `prompt_message_context.mentions`;
- typed reply target in `prompt_message_context.reply` or `reply_context`;
- recent scoped participants already present in `scope_users`;
- self-cognition or result-ready target scope for callback delivery.

`scope_users` is an internal brain-side lookup source only. It must not be
sent wholesale to adapters, added to the dialog prompt, or exposed in
`ChatResponse`. Only users whose exact authored token appears in outbound text
may become `delivery_mentions` candidates.

Candidate construction must exclude:

- the active character or bot account;
- users without `display_name`;
- users without `platform_user_id`;
- users whose display name is ambiguous within the candidate set;
- users whose exact `@display_name` token does not appear in the outbound
  text outside fenced code blocks.

### Token Matching Contract

- A token is the literal string `@` followed by the candidate `display_name`.
- Candidate `display_name` values are stripped for leading and trailing
  whitespace during candidate construction. Internal whitespace and punctuation
  remain unchanged.
- Matching is exact and case-sensitive. Do not normalize case, collapse
  whitespace, infer aliases, or match nicknames.
- Sort candidate display names by descending length before scanning, so longer
  names win over shorter names at the same position.
- Replace all non-overlapping valid occurrences outside fenced code blocks.
- Replacement applies across all valid candidates in the response.
- A valid occurrence must not be followed by a Unicode alphanumeric character
  or underscore. This prevents `@Alex` from matching the prefix of
  `@Alexandra`.
- If two or more candidates have the same stripped `display_name`, omit every
  candidate with that display name for the current response.
- If a display name is empty after stripping, omit it.
- Triple-backtick fenced code blocks are the only protected spans in this
  plan. Text outside fenced blocks is eligible for replacement.
- If a candidate appears only inside fenced code blocks, omit that candidate
  from `delivery_mentions`.

### Adapter Rendering

Discord:

```text
input text: " @Alex please look"
candidate: {"display_name": "Alex", "platform_user_id": "2787858400"}
output text: " <@2787858400> please look"
```

NapCat QQ:

```text
input text: "@Alex please look"
candidate: {"display_name": "Alex", "platform_user_id": "123456"}
output OneBot segments: [{"type": "at", "data": {"qq": "123456"}}, ...]
```

Rendering failure policy:

- invalid candidate: skip it;
- no exact token: leave text unchanged;
- ambiguous token: leave text unchanged;
- fenced-code-block occurrence: leave that occurrence unchanged;
- unsupported channel type or DM: existing adapter channel policy decides
  whether native mention rendering is allowed.
- Brain-owned producers require non-empty `platform_user_id`. Adapters still
  own platform-specific renderability checks, such as numeric ID validation for
  Discord and QQ, and must skip unrenderable candidates without blocking text
  delivery.

## LLM Call And Context Budget

No new LLM calls are added.

| Stage | Before | After | Response path | Context impact |
|---|---:|---:|---|---|
| Dialog generator | 1 call when selected text surface runs | 1 call | yes | Prompt wording changes only; no new dynamic payload required. |
| L3 content plan agent | 1 call when selected text surface runs | 1 call | yes | No required change in this plan. |

Default context cap: 50k tokens.

Prompt edits must not add platform names, adapter mechanics, database
concepts, candidate lists, or delivery feasibility inputs to the dialog prompt.
The prompt change should be shorter than the removed `mention_target_user`
contract wording.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/brain_service/delivery_mentions.py`
  - Owns bounded candidate construction, ambiguity filtering, fenced-block
    masking, and exact token detection for brain-owned outbound producers.
- `src/adapters/inline_mentions.py`
  - Owns shared platform-neutral inline token parsing helpers used by Discord
    and NapCat rendering.
- `tests/test_inline_delivery_mentions.py`
  - Covers candidate filtering, duplicate display-name omission, exact token
    matching, fenced-block skipping, and stale-field exclusion.

### Modify

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Remove the `@` prohibition.
  - Instruct dialog to write `@display_name` when it intentionally tags a
    current user or named participant already present in upstream dialog
    directives.
  - Do not add candidate lists, `scope_users`, platform ids, or delivery
    feasibility data to the dialog prompt.
  - Remove `mention_target_user` from active prompt output contract and parser
    expectations.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Preserve the already-built `scope_users` list in the final graph result so
    service response assembly can use it as a brain-side candidate lookup.
  - Do not add `scope_users` to any prompt or adapter-facing response.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Remove active `mention_target_user` schema usage.
  - Keep `scope_users` as internal graph state only.
- `src/kazusa_ai_chatbot/state.py`
  - Remove `mention_target_user` from the active dialog state shape after
    service, self-cognition, and tests stop consuming it.
- `src/kazusa_ai_chatbot/service.py`
  - Replace `mention_target_user` prefix metadata assembly with inline
    candidate assembly for normal `/chat`.
  - Replace background result-ready prefix metadata with inline candidate
    assembly over result-ready target scope.
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - Stop generating prefix metadata from `mention_target_user`.
  - Build inline candidates from self-cognition target scope only when the
    candidate text contains exact `@display_name`.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Stop passing `mention_target_user` through self-cognition action tracking.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Continue forwarding candidate lists from action candidates unchanged.
- `src/kazusa_ai_chatbot/self_cognition/delivery.py`
  - Update docstrings and expectations from mention metadata to inline render
    candidates.
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`
  - Preserve forwarding behavior and update validation/docstrings to inline
    render candidates.
- `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py`
  - Update protocol docstrings.
- `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`
  - Update docstrings while preserving payload shape.
- `src/adapters/discord_adapter.py`
  - Replace `_outbound_text_with_delivery_mentions` prefix behavior with
    inline replacement.
  - Apply replacement in both normal `/chat` sends and runtime callback sends.
- `src/adapters/napcat_qq_adapter/outbound.py`
  - Replace `prefix_user_mention_segments` with inline segment rendering.
- `src/adapters/napcat_qq_adapter/ws_adapter.py`
  - Continue passing group delivery candidates into outbound payload builders.
- `src/adapters/napcat_qq_adapter/runtime_api.py`
  - Preserve request shape and update docstrings if present.
- `src/adapters/README.md`
  - Document adapter-owned inline rendering and no-op fallback.
- `src/kazusa_ai_chatbot/brain_service/README.md`
  - Redefine `delivery_mentions` as inline render candidates.
- `src/kazusa_ai_chatbot/dispatcher/README.md`
  - Remove prefix-only language.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Remove `mention_target_user` delivery wording and prefix examples.
- `tests/test_dialog_agent.py`, `tests/test_dialog_mention_target_user.py`
  - Replace `@` prohibition and `mention_target_user` prompt assertions with
    inline `@display_name` prompt contract tests.
- `tests/test_dialog_inline_mentions_live_llm.py`,
  `tests/test_dialog_l3_surface_contract_live_llm.py`,
  `tests/test_dialog_one_bubble_layout_live_llm.py`
  - Replace live/output-shape assumptions about `mention_target_user` with the
    confirmed inline mention prompt-shape contract.
- `tests/test_persona_supervisor2.py`, `tests/test_l2d_l3_surface_handoff.py`,
  `tests/test_rag_dialog_event_logging.py`,
  `tests/test_self_cognition_integration.py`
  - Update graph, handoff, event-log, and integration fixtures for removed
    `mention_target_user` output and retained internal `scope_users`.
- `tests/test_runtime_adapter_registration.py`
  - Replace prefix tests with Discord and NapCat inline replacement tests for
    normal, runtime callback, and unsupported/no-op sends.
- `tests/test_service_background_consolidation.py`,
  `tests/test_delivery_mentions.py`, `tests/test_self_cognition_tracking.py`,
  `tests/test_background_work_delivery.py`,
  `tests/test_background_artifact_delivery.py`
  - Replace service, self-cognition, dispatcher, and result-ready prefix
    metadata assertions with inline candidate assertions.
- `tests/test_control_console_kazusa_client.py`
  - Keep count projection behavior stable and update sample payloads to the new
    candidate shape.

### Keep

- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - Keep `ChatResponse.delivery_mentions` field name and list type unchanged.
- `src/kazusa_ai_chatbot/message_envelope/*`
  - Keep inbound mention normalization unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`
  - Keep `use_reply_feature` ownership unchanged.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - Keep unchanged. If verification proves L3 explicitly blocks
    `@display_name`, stop and update this plan before touching L3.
- `src/control_console/kazusa_client.py`
  - Keep count-only projection behavior unchanged. Update tests or sample
    payload fixtures only when the candidate shape changes their inputs.

### Delete

- Delete active prompt/test expectations that require `mention_target_user`.
- Delete prefix-specific adapter helpers after inline replacements cover all
  call sites.
- Do not delete public `delivery_mentions` fields or runtime request fields.

## Overdesign Guardrail

- Actual problem: Dialog-authored `@display_name` text is not converted into
  native platform mentions, while the existing prefix-only mechanism hides the
  decision behind `mention_target_user`.
- Minimal change: Keep `delivery_mentions` as the existing list field, change
  item semantics to minimal inline render candidates, train dialog to author
  tag signs, and let adapters perform exact platform replacement.
- Ownership boundaries: dialog owns visible wording; service owns bounded
  candidate construction; adapters own native rendering; persistence stores
  platform-neutral authored text.
- Rejected complexity: no new response field, no adapter DB lookup, no fuzzy
  matching, no alias resolver, no whole-channel roster, no platform prompt
  assumptions, no compatibility shim, no feature flag, no role/channel/everyone
  mention support.
- Evidence threshold: add a new schema field, alias matching, role mentions, or
  broader roster disclosure only after an observed production case proves exact
  `@display_name` matching is insufficient and the user approves that expanded
  contract.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate cutover
  strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the target modules as
  high-scrutiny changes. Updating an existing module outside the listed change
  surface requires stopping and updating this plan before implementation.
- The responsible agent may create only the two helper modules named in
  `Change Surface`. Additional helpers, wrappers, aliases, task managers, or
  abstractions are out of scope.
- The responsible agent must search for existing equivalent behavior before
  adding helper code. If equivalent behavior already exists, move or reuse it
  instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad prompt rewrites, adapter modularization, or
  sequence-delivery work.
- If this plan and `dialog_message_sequence_delivery_plan.md` disagree,
  preserve this plan's mention semantics and report the discrepancy before
  executing overlapping steps.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent updates focused candidate-construction tests.
   - File: `tests/test_inline_delivery_mentions.py`.
   - Add tests for exact token detection, no token no candidate, duplicate
     display name omission, missing platform id omission, active character
     omission, and fenced-block skipping.
   - Expected pre-implementation result: fails because the helper module does
     not exist.
2. Parent updates adapter inline rendering tests.
   - File: `tests/test_runtime_adapter_registration.py`.
   - Replace prefix tests with Discord inline replacement tests.
   - Replace prefix tests with NapCat inline segment tests.
   - Include stale prefix-shaped metadata as ignored input.
   - Expected pre-implementation result: fails because adapters still prefix or
     ignore inline tokens.
3. Parent updates dialog prompt contract tests.
   - File: `tests/test_dialog_agent.py`.
   - Assert the prompt allows `@display_name` for intentional tags and does not
     require `mention_target_user`.
   - Expected pre-implementation result: fails because the prompt forbids `@`.
4. Parent updates service/self-cognition/background tests.
   - Files: `tests/test_service_background_consolidation.py`,
     `tests/test_delivery_mentions.py`, `tests/test_self_cognition_tracking.py`,
     `tests/test_background_work_delivery.py`,
     `tests/test_background_artifact_delivery.py`.
   - Replace prefix metadata expectations with inline candidate expectations.
   - Expected pre-implementation result: fails because producers still emit
     prefix metadata from `mention_target_user`.
5. Record focused failing tests or baseline behavior before production edits.
6. Create
   `src/kazusa_ai_chatbot/brain_service/delivery_mentions.py`.
   - Implement candidate normalization, ambiguity filtering, fenced-block
     masking, and exact token matching.
   - Keep the helper independent of adapter platform syntax.
7. Create `src/adapters/inline_mentions.py`.
   - Implement shared exact-token scanning for adapter renderers.
   - Keep platform-native rendering outside this shared helper.
8. Update `dialog_agent.py`.
   - Remove `@` prohibition.
   - Add concise platform-neutral instruction to use `@display_name` when
     intentionally tagging the current user or a named participant already
     present in upstream dialog directives.
   - Remove active `mention_target_user` output contract and parsing dependency.
9. Update service and graph state wiring.
   - Expose bounded `scope_users` to service response assembly.
   - Replace normal `/chat` prefix metadata generation with inline candidate
     assembly.
   - Replace background result-ready prefix metadata generation with inline
     candidate assembly.
10. Update self-cognition and dispatcher docs/code.
    - Stop generating mention candidates from `mention_target_user`.
    - Preserve dispatcher forwarding shape.
11. Update Discord adapter rendering.
    - Replace prefix rendering with inline exact replacement.
    - Apply the same semantics to normal and runtime callback sends.
12. Update NapCat QQ adapter rendering.
    - Replace prefix rendering with inline segment generation.
    - Preserve reply segment behavior when present.
13. Update ICDs and subsystem READMEs listed in `Change Surface`.
14. Parent runs focused tests and prompt render checks.
15. Parent runs static greps for legacy prefix and `mention_target_user`
    delivery references.
16. Parent runs broader regression tests listed in `Verification`.
17. Parent runs live LLM checks one case at a time only after deterministic
    tests pass and records a `debug-llm` review artifact.
18. Run the fallback code-review gate after planned verification passes.
19. Remediate review findings inside the approved change surface and rerun
    affected verification.

## Execution Model

- Parent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent establishes the focused test contract and records expected failure or
  baseline before production implementation starts.
- Single-agent fallback execution owns both tests and production edits.
- After planned verification, the same agent must switch to review posture,
  inspect the full diff against this plan, and record findings and fixes.

## Progress Checklist

For every stage, record verification output and sign-off as
`<agent/date>` in `Execution Evidence` before moving to the next stage.

- [x] Stage 1 - focused mention contract tests established
  - Covers steps 1-4; verify focused tests and record expected failures or
    current baseline behavior.
- [x] Stage 2 - brain mention candidate construction implemented
  - Covers steps 6 and 9; verify `tests/test_inline_delivery_mentions.py` and
    affected service tests pass.
- [x] Stage 3 - dialog prompt contract updated
  - Covers step 8; verify dialog prompt tests and prompt render check pass.
- [x] Stage 4 - adapter inline rendering implemented
  - Covers steps 7, 11, and 12; verify Discord and NapCat inline rendering
    tests pass.
- [x] Stage 5 - callback, self-cognition, docs, and regression verification complete
  - Covers steps 10, 13, 14, 15, 16, and 17; verify greps, focused tests,
    regressions, and the live LLM review artifact after confirmation.
- [x] Stage 6 - fallback code review complete
  - Covers steps 18-19; verify review completion, remediations, rerun commands,
    and residual risks before marking the plan completed.

## Verification

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests/test_inline_delivery_mentions.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_dialog_agent.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_on_message_replaces_inline_delivery_mentions -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_discord_runtime_send_replaces_inline_delivery_mentions -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_handle_event_replaces_inline_delivery_mentions -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_runtime_send_replaces_inline_delivery_mentions -q`
- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_unsupported_adapter_leaves_inline_mentions_plain -q`

### Regression Tests

- `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_service_background_consolidation.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_delivery_mentions.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_self_cognition_tracking.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_background_work_delivery.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_background_artifact_delivery.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_control_console_kazusa_client.py -q`

### Prompt Render Checks

- Render `_DIALOG_GENERATOR_PROMPT` through the existing test helper or script.
- Expected result: prompt renders without `.format(...)` placeholder errors,
  allows intentional `@display_name`, and contains no platform-specific native
  mention syntax.

### Static Greps

- `rg "mention_target_user" src/kazusa_ai_chatbot src/adapters tests`
  - Expected result: no active delivery contract references remain. Historical
    archive files are outside this grep. Any remaining production match must be
    reviewed and justified before sign-off.
- `rg "\"placement\"|\"requested_by\"" src/kazusa_ai_chatbot src/adapters tests`
  - Expected result: no active `delivery_mentions` producer or assertion uses
    these fields.
- `rg "prefix mention|prefix-only|placement.*prefix" src/kazusa_ai_chatbot src/adapters tests`
  - Expected result: no active delivery mention contract describes prefix
    placement.
- `rg "Discord|NapCat|OneBot|QQ|CQ" src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Expected result: no matches introduced by this plan.

### Live LLM Quality Gate

Do not create or edit the live LLM test file until the user confirms this test
set. After confirmation and implementation, run each test one at a time with
`-q -s`, inspect the emitted evidence, and author a `debug-llm` review
artifact.

Create `tests/test_dialog_inline_mentions_live_llm.py` with these cases:

- `test_dialog_inline_mention_shape_current_user_live_llm`
  - Input: synthetic group reply state with `user_name="Alex"` and dialog
    directives that explicitly make a visible tag natural.
  - Hard gates: raw dialog-generator JSON parses; `final_dialog` is
    `list[str]`; raw output has no `mention_target_user`; no platform names or
    native mention syntax appear.
  - Contract gate: at least one final dialog item contains exact `@Alex`.
- `test_dialog_inline_mention_shape_no_tag_live_llm`
  - Input: synthetic group reply state with the same current user but no tag
    directive.
  - Hard gates: same JSON shape and forbidden-field checks as above.
  - Contract gate: final dialog contains no `@` token.
- `test_dialog_inline_mention_shape_named_participant_live_llm`
  - Input: synthetic group state whose upstream dialog directives already name
    `Moca` and instruct visible tagging of that named participant.
  - Hard gates: same JSON shape and forbidden-field checks as above.
  - Contract gate: at least one final dialog item contains exact `@Moca`.
- `test_dialog_inline_mention_shape_multiple_tags_live_llm`
  - Input: synthetic group state whose upstream dialog directives already name
    `Alex` and `Moca` and instruct visible tagging of both participants.
  - Hard gates: same JSON shape and forbidden-field checks as above.
  - Contract gate: final dialog contains exact `@Alex` and exact `@Moca`.
- `test_dialog_inline_mention_shape_fenced_at_live_llm`
  - Input: synthetic technical reply state whose directives include a fenced
    code example containing an `@` character.
  - Hard gates: same JSON shape and forbidden-field checks as above.
  - Contract gate: any `@` inside the fenced block remains literal text; the
    model does not invent platform/native mention syntax.

Run each case individually with:
`venv\Scripts\python.exe -m pytest tests/test_dialog_inline_mentions_live_llm.py::<test_name> -q -s`

The review artifact must include each case's prompt identity or version, raw
model output, parsed output, hard-gate result, and quality notes about whether
the model followed the tag-sign contract.

## Independent Code Review

Run this fallback review gate after all `Verification` commands pass and before
final sign-off. The user explicitly approved execution without subagents, so
the same agent must switch to review posture and inspect the full diff.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Ownership boundaries: dialog owns visible tag text, service owns bounded
  candidates, adapters own native rendering, and persistence remains
  platform-neutral.
- Hidden fallback risks: no prefix fallback, no `mention_target_user`
  compatibility shim, no adapter DB lookup, no fuzzy matching, no leaked
  `global_user_id`, no platform-specific dialog prompt assumptions.
- Regression and handoff quality: focused tests map to actual risks, static
  grep expectations are current, the message-sequence plan overlap is
  documented, and deferred sequence delivery work is not silently implemented.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a finding requires a new response field,
adapter database access, fuzzy matching, role mentions, sequence delivery, or a
broader delivery contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Dialog prompt and tests allow intentional `@display_name` tags.
- Active dialog output handling no longer depends on `mention_target_user`.
- Normal `/chat` responses expose only minimal inline mention candidates in
  `delivery_mentions`.
- `delivery_mentions` candidates contain no `placement`, `global_user_id`, or
  `requested_by` fields.
- Candidate construction omits ambiguous, unrenderable, inactive, and
  not-present users.
- Discord normal sends and runtime callback sends replace exact inline tags
  with native user mentions.
- NapCat QQ normal sends and runtime callback sends replace exact inline tags
  with OneBot `at` segments.
- Debug or unsupported adapters can ignore `delivery_mentions` and still send
  readable text.
- Fixed-format blocks are not corrupted by mention replacement.
- Dispatcher, self-cognition, background result-ready, Brain Service, Adapter,
  and Dispatcher docs describe inline render candidates instead of prefix
  mentions.
- Focused tests, regression tests, static greps, prompt render checks, and the
  independent code review gate pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Wrong user tagged because display names collide | Omit ambiguous display names from candidates | Candidate duplicate-name tests |
| Adapter leaks internal identity metadata | Emit only `display_name` and `platform_user_id` | Service tests and static grep for removed fields |
| Dialog stops tagging because local LLM avoids `@` | Use short positive prompt instruction and live LLM quality gate | Dialog prompt tests and live LLM artifact |
| Native replacement corrupts code or JSON | Skip fenced code blocks | Fenced-block unit tests |
| Sequence plan reintroduces prefix behavior | This plan owns mention semantics and static greps reject prefix wording | Plan overlap review and static greps |
| Adapter silently cannot render nonnumeric IDs | Adapter skips invalid candidates and sends authored text unchanged | Adapter invalid-candidate tests |
| Persistence stores native syntax | Keep rendering adapter-local and do not write native syntax back to body text | Service persistence regression tests |

## Execution Evidence

- Focused test baseline: failing red cases observed for missing helper, adapter inline rendering, dialog contract, service delivery_mentions, self-cognition inline candidates, and embedded-token boundary before implementation/remediation.
- Production implementation summary: added brain inline candidate builder, shared adapter inline scanner, Discord/NapCat inline rendering, minimal delivery_mentions shape, dialog prompt/output cleanup, service/background/self-cognition candidate assembly, and README/ICD updates.
- Static grep results: `rg` gates for `mention_target_user`, legacy prefix placement, `"placement"`, `"requested_by"`, and platform-native prompt terms returned no active code/test matches on 2026-07-01.
- Prompt render checks: `tests/test_dialog_agent.py` passed after prompt update; live multiple-tag case initially exposed fenced raw JSON, then passed after adding the no-Markdown-fence output instruction.
- Focused test results: `test_inline_delivery_mentions.py` 7 passed; targeted adapter inline tests 7 passed; dialog-agent full deterministic file 22 passed; service/background/self-cognition focused producer tests passed.
- Regression test results: service background consolidation 27 passed; delivery/background files 15 passed; self-cognition tracking 42 passed; control console client 1 passed; runtime adapter registration 63 passed as one file after fixing the NapCat CLI test seam.
- Live LLM review artifact: `tests/test_dialog_inline_mentions_live_llm.py` five cases passed one at a time with traces under `test_artifacts/llm_traces/dialog_inline_mentions_live_llm__*.json`.
- Independent code review: same-agent fallback review found embedded `email@Alex` replacement risk and NapCat CLI test leakage into a real websocket reconnect loop; fixed with prefix-boundary checks plus an explicit CLI adapter-class patch seam, then reran affected tests.
- Residual risks: role/channel/broadcast mentions, fuzzy matching, adapter DB lookup, and sequence-delivery behavior remain out of scope.
