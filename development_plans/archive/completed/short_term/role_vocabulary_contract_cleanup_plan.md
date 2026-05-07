# role vocabulary contract cleanup plan

## Summary

- Goal: Remove ambiguity between conversation author roles, mention entity kinds, referent grammar roles, bot platform identity, assistant turns, and active character identity.
- Plan class: medium
- Status: completed
- Overall cutover strategy: bigbang for code/docs/tests; no database migration; no compatibility shim.
- Highest-risk areas: accidentally changing stored `conversation_history.role` semantics, weakening adapter-to-brain envelope validation, or adding prompt-facing vocabulary that makes the local LLM infer hidden schemas.
- Acceptance criteria: every role-like field has one documented semantic namespace; envelope mentions are no longer described as conversation roles; service and history helpers use `assistant` for authored bot output; tests prove no `role="bot"` conversation row is written.

## Context

The Stage 2 typed message envelope work made transport metadata first-class, but a follow-up survey found that the repo still uses the word `role` for several unrelated concepts:

- Conversation rows use `role="user" | "assistant"` as the author role.
- Message envelope mentions use `role="bot" | "user" | "role" | "channel" | "everyone" | "unknown"` to classify the mentioned platform entity.
- Decontextualizer referents use `role="subject" | "object" | "time"` to describe grammar role.
- Dispatcher scheduled tasks use `bot_role` for permission level.
- Local names such as `_save_bot_message`, `bot_output`, and "user-bot interaction" describe assistant-authored rows that are persisted as `role="assistant"`.

The survey did not find source code writing `role="bot"` as a conversation-history author role. The current problem is naming and type-boundary ambiguity, not observed persisted data corruption.

This matters because the system is intentionally local-LLM friendly. Dict-shaped payloads and prompts must not force a weaker model to infer which semantic namespace a generic `role` key belongs to.

## Mandatory Rules

- Preserve `conversation_history.role` as the stored author-role field with values `user` and `assistant`.
- Do not introduce `role="bot"` as a conversation-history author role.
- Do not migrate the MongoDB field `conversation_history.role`; this plan is not a database migration.
- Do not add compatibility shims, secondary role paths, or legacy fallback handling.
- Do not move platform normalization out of adapters or back into brain modules.
- Do not add deterministic semantic filters over user intent. This plan is vocabulary/type cleanup only.
- Keep local LLM prompts schema-explicit. If a prompt exposes role-like fields, the prompt must name the namespace plainly.
- Do not hard-code concrete user or character names in examples, docs, prompts, or tests added by this plan.
- Follow project Python style for `.py` edits: imports at top, complete docstrings for public helpers, no broad `except Exception` outside existing external-boundary code, no scattered defaults, no unnecessary helper abstractions.
- Keep edits narrowly scoped to role vocabulary, typing, docs, and tests listed in this plan.

## Must Do

- Add a role vocabulary section to `src/kazusa_ai_chatbot/message_envelope/README.md`.
- Introduce explicit type aliases for the distinct role namespaces:
  - `ConversationAuthorRole = Literal["user", "assistant"]`
  - `MentionEntityKind = Literal["bot", "user", "platform_role", "channel", "everyone", "unknown"]`
  - `ReferentRole = Literal["subject", "object", "time"]`
  - `BotPermissionRole = str` or a narrowed permission Literal if the existing dispatcher permission set is fully enumerated.
- Rename the envelope mention field from `role` to `entity_kind` in the brain-facing envelope contract.
- Update QQ and Discord adapter normalizers to emit `entity_kind`, not mention `role`.
- Update service input models and conversation document schemas so mention metadata uses `entity_kind`.
- Update queue, resolver, relevance, RAG/search projections, and history trimming consumers to read `entity_kind`.
- Rename local service/helper language where it describes assistant-authored output:
  - `_save_bot_message` -> `_save_assistant_message`
  - `bot_output` -> `assistant_output`
  - "bot rows" in comments/docstrings -> "assistant rows" when referring to `role="assistant"`
- Keep `platform_bot_id` as-is. It correctly means the platform account id for the bot account.
- Keep `bot_name` as-is only where the value comes from adapter/platform account metadata. Use `character_name` or `display_name` where the value is persona-facing rather than platform-account-facing. Before applying renames, enumerate every `bot_name` call site in `src/` and `tests/` and record a classification table in `Execution Evidence` (one row per site: file:line, current value source, classification, target name). Mid-stream interpretation is not allowed; the table must exist before any rename in this category lands.
- Update prompts that show `chat_history` examples so `role` is explicitly described as conversation author role, not generic role.
- Add tests proving mention entity kinds and conversation author roles cannot be confused.

## Deferred

- Do not rename the MongoDB `conversation_history.role` field to `author_role`.
- Do not backfill existing stored mention documents from `role` to `entity_kind` unless a later migration plan explicitly approves it.
- Do not redesign the message envelope beyond this vocabulary cleanup.
- Do not change relevance behavior, queue behavior, RAG retrieval behavior, or cognition semantics except for field-name updates required by this plan.
- Do not introduce a new normalizer package under `kazusa_ai_chatbot.message_envelope`; adapter-specific normalization remains in `src/adapters`.
- Do not rename public platform concepts such as `platform_bot_id` when they really refer to the platform bot account.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Source code mention field | bigbang | Change `Mention.role` to `Mention.entity_kind` and update all source consumers in one pass. |
| Stored conversation author role | compatible | Keep `conversation_history.role` unchanged as `user | assistant`. |
| Persisted mention documents | n/a | No production data exists yet for the affected fields; the new contract is the only contract. No read fallback is needed. |
| Tests and fixtures | bigbang | Update fixtures to the new mention field in the same change. |
| Prompt examples | bigbang | Update examples so the role namespace is explicit. |

Because the user has mandated strict interfaces and no shim layers, source code after this plan must not support both `mention.role` and `mention.entity_kind` as equal live paths.

## Agent Autonomy Boundaries

- The agent may choose local rename mechanics only when they preserve this plan's contracts.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, broad prompt rewrites, or database migrations.
- If a required rename reveals a persisted-data compatibility issue, the agent must stop and report it instead of silently adding fallback code.
- If the plan and code disagree, preserve the plan's stated semantic boundary and report the discrepancy.

## Target State

The repo uses four clear namespaces:

| Namespace | Field / Type | Values | Meaning |
|---|---|---|---|
| Conversation author | `ConversationMessageDoc.role`, `ConversationAuthorRole` | `user`, `assistant` | Who authored a stored conversation row. |
| Mention entity | `Mention.entity_kind`, `MentionEntityKind` | `bot`, `user`, `platform_role`, `channel`, `everyone`, `unknown` | What kind of platform entity was mentioned in the envelope. |
| Referent grammar | `ReferentResolution.referent_role`, `ReferentRole` | `subject`, `object`, `time` | Grammatical role of an unresolved/resolved phrase. |
| Bot permission | `DispatchContext.bot_permission_role`, `BotPermissionRole` | dispatcher permission names | Permission level for scheduled/tool dispatch. |

`assistant` is the only stored author value for character-generated conversation output. `bot` is reserved for platform-account concepts and mention entity classification.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stored author role | Keep `role="assistant"` | It matches chat/LLM convention and current storage behavior. |
| Mention field name | Rename to `entity_kind` | A mention's `role` is not a conversation role. |
| Discord role mention value | Rename value from `role` to `platform_role` | Avoid the confusing shape `mention.role == "role"`. |
| Bot terminology | Keep only for platform account concepts | `bot` maps to transport identity; `character` maps to persona identity; `assistant` maps to authored output. |
| Database migration | Do not migrate `conversation_history.role` | No corruption was found; a storage rename would add risk without fixing the immediate ambiguity. |
| Backward compatibility | No live dual path | The adapter and brain are updated together; the brain may fail on non-contract envelopes. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/message_envelope/types.py`: rename mention field and type aliases.
- `src/kazusa_ai_chatbot/message_envelope/resolvers.py`: resolve bot mentions via `entity_kind`.
- `src/kazusa_ai_chatbot/message_envelope/README.md`: add formal role vocabulary and update mention schema.
- `src/adapters/discord_adapter.py`: emit `entity_kind`; emit `platform_role` for Discord role mentions.
- `src/adapters/napcat_qq_adapter.py`: emit `entity_kind`.
- `src/kazusa_ai_chatbot/service.py`: narrow input model field; rename assistant-save helper/local variables.
- `src/kazusa_ai_chatbot/db/schemas.py`: clarify `ConversationMessageDoc.role`; update mention schema to `entity_kind`.
- `src/kazusa_ai_chatbot/utils.py`: update docstrings from bot rows to assistant rows where applicable.
- `src/kazusa_ai_chatbot/chat_input_queue.py`: read `mention.entity_kind == "bot"`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: rename referent `role` to `referent_role`.
- Referent consumers under `src/kazusa_ai_chatbot/nodes`: update schema examples, validation, and prompt wording.
- `src/kazusa_ai_chatbot/dispatcher`: rename in-code variables and dataclass fields from `bot_role` to `bot_permission_role`. Any persisted/serialized scheduled-event documents keep `bot_role` as the wire/storage key (same asymmetry as `conversation_history.role`); the translation lives at the serialization boundary only. Do not migrate stored documents and do not add a runtime dual-read fallback.
- Tests under `tests/` that construct mentions, referents, chat history, or service requests.

### Keep

- `ConversationMessageDoc.role` storage field and values.
- `platform_bot_id`.
- Adapter ownership of platform normalization.
- Message envelope as a TypedDict contract.

### Create

- No new production module is required.
- New tests may be added to existing test files if focused files become too large.

## Implementation Order

1. Update the README role vocabulary first so the code edits have a written contract.
2. Update `message_envelope.types` and schemas to define the new names.
3. Update adapter normalizers to produce the new mention shape.
4. Update service intake, queue, resolver, history, and RAG/search projections to consume the new mention shape.
5. Update referent schema from `role` to `referent_role` and adjust prompts/tests.
6. Enumerate every `bot_name` call site in `src/` and `tests/`, classify each as platform-account (keep) or persona-facing (rename to `character_name` or `display_name`), and record the table in `Execution Evidence` before applying any rename in this category.
7. Rename assistant-output service helper and docstrings, and apply the `bot_name` renames recorded in step 6.
8. Update tests and fixtures in one pass.
9. Run static greps to prove old live names are gone where required.
10. Run focused tests, then the full default suite.

## Progress Checklist

- [x] Stage A — vocabulary contract documented
  - Covers: README role vocabulary and final namespace table.
  - Verify: `rg -n "MentionRole|entity_kind|platform_role|ConversationAuthorRole" src\kazusa_ai_chatbot\message_envelope\README.md`.
  - Evidence: record changed doc sections and grep output in `Execution Evidence`.
  - Handoff: next stage starts with type/schema edits.
  - Sign-off: `Codex / 2026-05-01` after verification and evidence are recorded.

- [x] Stage B — mention entity field renamed
  - Covers: `message_envelope.types`, adapters, resolver, service input models, schemas, queue, projections, tests.
  - Verify: `rg -n "mention\.role|\"role\": \"bot\"|\"role\": \"role\"|MentionRole" src tests` returns no live mention-contract matches; allowed `role="user|assistant"` history fixtures may remain.
  - Evidence: record grep output and focused envelope/adapter test output.
  - Handoff: next stage starts with referent namespace cleanup.
  - Sign-off: `Codex / 2026-05-01` after verification and evidence are recorded.

- [x] Stage C — referent grammar field disambiguated
  - Covers: `ReferentResolution`, referent validation, decontextualizer/cognition prompt examples, referent tests.
  - Verify: `rg -n '"role": "subject"|"role": "object"|"role": "time"|ReferentResolution.*role' src tests` returns no obsolete referent-schema matches.
  - Evidence: record focused referent/decontextualizer/cognition test output.
  - Handoff: next stage starts with assistant/bot naming cleanup.
  - Sign-off: `Codex / 2026-05-01` after verification and evidence are recorded.

- [x] Stage D — assistant-vs-bot naming cleanup complete
  - Covers: `_save_assistant_message`, local variable renames, docstrings/comments in scoped history and service code.
  - Verify: `rg -n "_save_bot_message|bot_output|bot rows|user-bot interaction" src tests` returns no obsolete assistant-output naming except allowed platform-account contexts.
  - Evidence: record grep output and service/history tests.
  - Handoff: next stage starts with final verification.
  - Sign-off: `Codex / 2026-05-01` after verification and evidence are recorded.

- [x] Stage E — final verification and acceptance
  - Covers: static greps, focused tests, full default suite.
  - Verify: all commands in `Verification` pass.
  - Evidence: record exact command output in `Execution Evidence`.
  - Handoff: plan may be marked completed after owner approval.
  - Sign-off: `Codex / 2026-05-01` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg -n "mention\.role|MentionRole|\"role\": \"bot\"|\"role\": \"role\"" src tests`
  - Allowed only in historical comments if explicitly marked non-live; otherwise no matches.
- `rg -n '"role": "subject"|"role": "object"|"role": "time"' src tests`
  - Must return no obsolete referent-schema matches.
- `rg -n "_save_bot_message|bot_output|bot rows|user-bot interaction" src tests`
  - Must return no obsolete assistant-output naming.
- `rg -n '"role": "assistant"|"role": "user"' src tests`
  - Must show only conversation-author contexts.
- `rg -n "platform_bot_id" src tests`
  - Matches are allowed because this is platform account identity.

### Tests

- `python -m pytest tests/test_message_envelope.py tests/test_adapter_envelope_normalizers.py tests/test_service_input_queue.py -q`
- `python -m pytest tests/test_conversation_history_envelope.py tests/test_build_interaction_history_recent.py tests/test_relevance_agent.py -q`
- `python -m pytest tests/test_referent_resolution.py tests/test_decontexualizer_referents.py tests/test_cognition_clarification_consumers.py -q`
- `python -m pytest -q`

### Compile

- `python -m py_compile` over all changed Python files.

## Acceptance Criteria

This plan is complete when:

- `conversation_history.role` remains the only generic `role` field in conversation history and only means `user | assistant`.
- Envelope mentions use `entity_kind`, not `role`.
- Discord platform role mentions are represented as `platform_role`, not `role`.
- Referents use `referent_role`, not generic `role`.
- Service code writes assistant rows through assistant-named helpers and variables.
- The type aliases and renamed contracts make `role="bot"` as a conversation author role unreachable by construction. The survey shows no such writes exist today; this acceptance line is forward prevention, not a bug fix.
- The message-envelope README documents all role-like namespaces and ownership boundaries.
- Focused tests and full default pytest pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Hidden tests or fixtures still construct `mentions[].role` | Bigbang fixture update and static grep | Envelope/adapter/service tests plus grep. |
| Prompt examples still teach generic `role` ambiguity | Rename prompt-facing referent field and add author-role wording | Prompt contract tests and greps. |
| Dispatcher permission rename touches persisted scheduled events | In-code rename to `bot_permission_role`; persisted serialization key stays `bot_role` with translation at the persistence boundary only. No DB migration. | Dispatcher tests plus a round-trip serialize/deserialize test for persisted scheduled events. |
| Agent adds compatibility fallback despite strict interface | Mandatory no-shim rule and greps for old field reads | Static grep for old field accesses. |

## LLM Call And Context Budget

This plan must not add any response-path or background LLM calls.

Prompt text may change only where examples or schema field names currently expose ambiguous role vocabulary. The expected context impact is neutral to slightly smaller because schema wording becomes more explicit and no new examples are required.

Verification must use existing deterministic tests. Live LLM tests are not required unless prompt edits break existing live-test contracts.

## Rollback / Recovery

- Code rollback path: revert the code changes for this plan as one unit.
- Data rollback path: none required; this plan does not migrate data.
- Irreversible operations: none.
- Required backup: none beyond normal repository version control.
- Recovery verification: rerun focused envelope/service/referent tests after rollback.

## Execution Evidence

- Stage A vocabulary grep: `rg -n "MentionRole|entity_kind|platform_role|ConversationAuthorRole" src\kazusa_ai_chatbot\message_envelope\README.md` returned README lines for `ConversationAuthorRole`, `MentionEntityKind`, `entity_kind`, and `platform_role`.
- Stage A diff check: `git diff --check -- src\kazusa_ai_chatbot\message_envelope\README.md` passed.
- `bot_name` call-site classification before rename:

| Site | Current value source | Classification | Target |
|---|---|---|---|
| `tests/test_bot_side_addressing.py:40` | assistant-save fixture display name | persona-facing character display name | rename key/value to `character_name` when service contract changes |
| `tests/test_bot_side_addressing.py:81` | assistant-save fixture display name | persona-facing character display name | rename key/value to `character_name` when service contract changes |
| `tests/test_e2e_live_llm.py:168` | `character_profile.get("name", _BOT_NAME)` | persona-facing character display name | rename state key to `character_name`; leave `_BOT_NAME` test fallback alone only if it is platform-account fixture text |
| `src/adapters/napcat_qq_adapter.py:236` | NapCat login account nickname | platform bot account metadata | keep `bot_name` |
| `src/adapters/napcat_qq_adapter.py:267` | NapCat login account nickname | platform bot account metadata | keep `bot_name` |
| `src/adapters/napcat_qq_adapter.py:373` | `get_login_info.nickname` | platform bot account metadata | keep `bot_name` |
| `src/adapters/napcat_qq_adapter.py:377` | NapCat fallback account label | platform bot account metadata | keep `bot_name` |
| `tests/test_persona_supervisor2.py:33` | graph-state fixture for active character display name | persona-facing character display name | rename state key to `character_name` |
| `src/kazusa_ai_chatbot/state.py:69` | brain graph state active character display name | persona-facing character display name | rename to `character_name` |
| `src/kazusa_ai_chatbot/service.py:409` | `_ensure_character_global_identity` parameter | persona-facing character display name | rename to `character_name` |
| `src/kazusa_ai_chatbot/service.py:416` | docstring for active character display name | persona-facing character display name | rename to `character_name` |
| `src/kazusa_ai_chatbot/service.py:424` | passed as character display name | persona-facing character display name | rename variable to `character_name` |
| `src/kazusa_ai_chatbot/service.py:447` | `result["bot_name"]` assistant-save payload | persona-facing character display name | rename payload key to `character_name` |
| `src/kazusa_ai_chatbot/service.py:461` | `_ensure_character_global_identity` call | persona-facing character display name | rename arg to `character_name` |
| `src/kazusa_ai_chatbot/service.py:470` | persisted assistant display name | persona-facing character display name | rename local variable to `character_name`; keep storage field `display_name` |
| `src/kazusa_ai_chatbot/service.py:646` | `_personality["name"]` | persona-facing character display name | rename local variable to `character_name` |
| `src/kazusa_ai_chatbot/service.py:652` | ensure character identity call | persona-facing character display name | rename arg to `character_name` |
| `src/kazusa_ai_chatbot/service.py:718` | graph state active character display name | persona-facing character display name | rename state key to `character_name` |
| `src/kazusa_ai_chatbot/service.py:752` | busy message fallback display name | persona-facing character display name | rename local variable to `character_name` |
| `src/kazusa_ai_chatbot/service.py:816` | busy message fallback display name | persona-facing character display name | rename local variable to `character_name` |
| `tests/test_relevance_agent.py:34` | relevance-state fixture display name | persona-facing character display name | rename state key to `character_name` if relevance still needs it |
| `tests/test_service_background_consolidation.py:429` | graph-state fixture display name | persona-facing character display name | rename state key to `character_name` |
| `tests/test_service_background_consolidation.py:473` | graph-state fixture display name | persona-facing character display name | rename state key to `character_name` |
| `tests/test_state.py:19` | state schema expectation | persona-facing character display name | rename to `character_name` |
| `tests/test_state.py:73` | state fixture display name | persona-facing character display name | rename to `character_name` |
| `tests/test_runtime_adapter_registration.py:213` | NapCat adapter account nickname fixture | platform bot account metadata | keep `bot_name` |
| `tests/test_runtime_adapter_registration.py:272` | NapCat adapter account nickname fixture | platform bot account metadata | keep `bot_name` |
| `src/kazusa_ai_chatbot/nodes/relevance_agent.py:269` | prompt variable for active character name | persona-facing character display name | rename template variable to `character_name` |
| `src/kazusa_ai_chatbot/nodes/relevance_agent.py:357` | prompt variable for active character name | persona-facing character display name | rename template variable to `character_name` |
| `src/kazusa_ai_chatbot/nodes/relevance_agent.py:383` | prompt variable for active character name | persona-facing character display name | rename template variable to `character_name` |
| `src/kazusa_ai_chatbot/nodes/relevance_agent.py:531` | `character_profile["name"]` | persona-facing character display name | rename format arg to `character_name` |
| `src/kazusa_ai_chatbot/nodes/relevance_agent.py:689` | manual harness state fixture | persona-facing character display name | rename state key to `character_name` |

- Static grep results:
- Stage E mention-contract grep: `rg -n 'mention\.role|MentionRole|"role": "bot"|"role": "role"' src tests` returned no matches.
- Stage E referent-schema grep: `rg -n '"role": "subject"|"role": "object"|"role": "time"' src tests` returned no matches.
- Stage E assistant-output naming grep: `rg -n '_save_bot_message|bot_output|bot rows|user-bot interaction' src tests` returned no matches.
- Stage E conversation-author role grep: `rg -n '"role": "assistant"|"role": "user"' src tests` returned matches only in conversation-author contexts: conversation-history storage rows, chat-history fixtures, prompt examples, and tests for those contracts.
- Stage E platform bot id grep: `rg -n 'platform_bot_id' src tests` returned only platform-account identity contexts.
- Stage E `bot_name` grep: `rg -n '\bbot_name\b' src tests` returned only NapCat adapter platform-account metadata and its adapter registration tests.
- Stage E dispatcher in-code permission grep: `rg -n '\.bot_role\b|bot_role=' src tests` returned no matches. Persisted scheduler document key `"bot_role"` remains intentionally at the serialization boundary.
- Stage E diff check: `git diff --check` passed with Git line-ending warnings only.
- Stage B mention-contract grep: `rg -n 'mention\.role|"role": "bot"|"role": "role"|MentionRole' src tests` returned no matches.
- Stage B compile: `python -m py_compile src\kazusa_ai_chatbot\message_envelope\types.py src\kazusa_ai_chatbot\message_envelope\__init__.py src\kazusa_ai_chatbot\message_envelope\resolvers.py src\adapters\envelope_common.py src\adapters\discord_adapter.py src\adapters\napcat_qq_adapter.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\db\schemas.py src\kazusa_ai_chatbot\chat_input_queue.py` passed with no output.
- Stage B focused tests: `python -m pytest tests\test_message_envelope.py tests\test_adapter_envelope_normalizers.py tests\test_service_input_queue.py tests\test_conversation_history_envelope.py tests\test_runtime_adapter_registration.py -q` passed: 38 passed in 3.07s.
- Stage C referent-schema grep: `rg -n '"role": "subject"|"role": "object"|"role": "time"|ReferentResolution.*role' src tests` returned no matches after the README role-vocabulary row was worded as `referents[].referent_role`.
- Stage C compile: `python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\nodes\referent_resolution.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_referent_resolution.py tests\test_decontexualizer_referents.py tests\test_msg_decontexualizer.py tests\test_cognition_clarification_consumers.py tests\test_cognition_referents_live_llm.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py` passed with no output.
- Stage C focused tests: `python -m pytest tests\test_referent_resolution.py tests\test_decontexualizer_referents.py tests\test_cognition_clarification_consumers.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_msg_decontexualizer.py -q` passed: 26 passed, 3 deselected in 4.31s.
- Stage D assistant-output naming grep: `rg -n '_save_bot_message|bot_output|bot rows|user-bot interaction' src tests` returned no matches.
- Stage D platform-account `bot_name` audit: `rg -n '\bbot_name\b' src tests` returned only NapCat adapter platform metadata and its adapter registration tests.
- Stage D compile: `python -m py_compile src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\dispatcher\__init__.py src\kazusa_ai_chatbot\dispatcher\task.py src\kazusa_ai_chatbot\dispatcher\tool_spec.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py tests\test_bot_side_addressing.py tests\test_state.py tests\test_persona_supervisor2.py tests\test_relevance_agent.py tests\test_service_background_consolidation.py tests\test_e2e_live_llm.py tests\test_dispatcher.py tests\test_scheduler_future_promise.py` passed with no output.
- Stage D focused tests: `python -m pytest tests\test_bot_side_addressing.py tests\test_state.py tests\test_persona_supervisor2.py tests\test_relevance_agent.py tests\test_service_background_consolidation.py tests\test_dispatcher.py tests\test_scheduler_future_promise.py tests\test_build_interaction_history_recent.py tests\test_conversation_history_envelope.py -q` passed: 60 passed, 4 deselected in 4.55s.
- Compile results: `python -m py_compile` over all changed and untracked Python files, excluding `experiments/`, passed with no syntax errors. Git emitted line-ending warnings only while enumerating changed files.
- Focused test results:
  - `python -m pytest tests/test_message_envelope.py tests/test_adapter_envelope_normalizers.py tests/test_service_input_queue.py -q` passed: 25 passed in 1.89s.
  - `python -m pytest tests/test_conversation_history_envelope.py tests/test_build_interaction_history_recent.py tests/test_relevance_agent.py -q` passed: 26 passed in 3.90s.
  - `python -m pytest tests/test_referent_resolution.py tests/test_decontexualizer_referents.py tests/test_cognition_clarification_consumers.py -q` passed: 12 passed, 3 deselected in 1.31s.
- Full test results: `python -m pytest -q` passed: 416 passed, 139 deselected in 6.67s.
- Notes / blockers: No blockers. Live LLM tests were not run for Stage E because the plan requires deterministic verification unless prompt edits break live-test contracts.
