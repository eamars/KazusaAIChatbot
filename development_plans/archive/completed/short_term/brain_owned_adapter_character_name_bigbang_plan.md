# brain owned adapter character name bigbang plan

## Summary

- Goal: make the active brain character profile the only semantic source of
  the character display name across every platform adapter and dispatcher
  delivery path.
- Plan class: large.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: bigbang.
- Highest-risk areas: registration/heartbeat schema drift, stale platform bot
  labels in mentions and replies, stale background source snapshots, and
  accidental staging of another agent's work.
- Acceptance criteria: every registered platform adapter uses the required
  brain-provided `character_name`; no runtime path uses a platform bot display
  name as character identity; focused and regression verification passes; the
  completed diff is reviewed and committed without unrelated worktree changes.

## Context

The active character profile is loaded by the brain service and already owns
cognition, dialog, and normal assistant persistence. Discord and NapCat
currently create a second name authority:

```text
platform bot nickname/display_name
  -> runtime adapter registration request
  -> remote adapter display-name fallback
  -> bot mention and reply display labels
```

This allows QQ id `3768713357`, or the equivalent Discord bot account, to be
presented to the brain under an old platform nickname after the active brain
profile has changed.

The completed
`development_plans/archive/completed/short_term/qq_adapter_readable_mentions_plan.md`
established readable adapter mention labels and explicitly selected the
platform bot name. This plan supersedes only that bot-name ownership decision.
Human participant, role, and channel labels remain platform-owned.

The active brain profile means the process-local profile loaded and validated
at service startup. An explicit profile maintenance operation becomes active
after the brain process reloads that profile. Registration and heartbeat
distribute the name the running brain currently uses.

The worktree contains another agent's changes in top-level documentation,
control-console, cognition, configuration, and tests. Those changes are
outside this plan. Target files were clean at discovery except
`development_plans/README.md`, whose existing hunks must be preserved.

## Mandatory Skills

- `development-plan`: govern planning, review, execution evidence, lifecycle,
  and sign-off.
- `local-llm-architecture`: preserve the adapter/brain semantic boundary and
  avoid new response-path calls or prompt changes.
- `py-style`: govern every Python source and test edit.
- `cjk-safety`: govern edits to Python tests containing Chinese or Japanese
  character-name fixtures and require immediate syntax verification.
- `test-style-and-execution`: govern deterministic test-first execution and
  regression verification.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python and pytest commands.
- Use `apply_patch` for manual file edits.
- Do not read `.env`.
- Do not use subagents. The user explicitly approved single-agent fallback
  execution and required parent-only plan and code review.
- Inspect the current hash or diff of each target before editing it. If a
  target acquires an unrecognized concurrent change, preserve it and reconcile
  only when the ownership boundary remains clear.
- Stage only this plan's paths and, for already-dirty shared files, only this
  plan's exact hunks. Do not stage another agent's changes.
- The active brain profile is the only source of character display-name
  semantics. Platform account ids remain adapter-owned transport identity.
- Human sender, human mention, role, and channel labels remain adapter-owned.
- Do not add a platform-name fallback, compatibility alias, dual request
  shape, optional legacy field, feature flag, or per-message name lookup.
- Do not add or change any LLM call, prompt, RAG stage, cognition stage, dialog
  stage, model route, or context budget.
- Preserve historical conversation rows and historical provenance snapshots.
  Do not run or create a conversation-history database migration.
- When executing pending/background output, use the current active brain name
  for new visible output and new assistant rows. Retained historical
  `source_character_name` fields are provenance only.
- A registration response without a non-empty string `character_name` is a
  contract failure. Startup registration must fail; heartbeat processing logs
  the failure and retains only the last successfully validated brain value.
- After automatic context compaction, reread this entire plan before
  continuing.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Before lifecycle completion, commit, or sign-off, perform the plan's
  parent-only Independent Code Review and record the result.

## Must Do

- Replace the runtime adapter registration request schema with a strict
  transport-only request containing required `platform_bot_id` and no
  `display_name`.
- Add required `character_name` to the registration and heartbeat response.
- Return the active brain profile name through both registration endpoints.
- Parse and validate the same response contract in Discord and NapCat.
- Require both platform normalizers to receive `character_name`.
- Override platform bot mention and reply labels with `character_name`.
- Remove NapCat `bot_name` semantic state and ignore the nickname returned by
  `get_login_info`.
- Ignore Discord's bot `display_name` for semantic body, mention, reply, and
  registration data.
- Remove dispatcher fallbacks to adapter `display_name`, adapter `bot_name`,
  and the generic `assistant` label.
- Require dispatcher contexts to carry brain-owned platform bot id and current
  character name.
- Use the current active brain name for accepted-task result delivery and
  self-cognition delivery while retaining old source snapshots as provenance.
- Canonicalize brain-side reply display for historical assistant rows by
  matching `reply_to_platform_user_id` to the current request's
  `platform_bot_id`.
- Update shared ICDs and focused deterministic tests.
- Run every verification gate and commit only this plan's completed changes.

## Deferred

- Do not implement live character-profile reload or a profile-management API.
- Do not change how human platform display names are selected or cached.
- Do not change raw platform ids, global character id resolution, CQ/Discord
  parsing grammar, outbound native mention rendering, or delivery receipts.
- Do not rewrite persisted historical conversation text, assistant
  `display_name`, memory, embeddings, or scheduled provenance.
- Do not modify top-level `README.md`, `README_CN.md`, or `docs/HOWTO.md`
  because their current changes belong to another agent and subsystem ICDs
  fully own this contract.
- Do not modify control-console, cognition, RAG, dialog, consolidation,
  character-profile storage, or model-route code.
- Do not add a brain call before each inbound platform event.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Registration request | bigbang | Remove `display_name`; require `platform_bot_id`; reject extra legacy fields. |
| Registration response | bigbang | Require non-empty `character_name` on register and heartbeat. |
| Discord | bigbang | Replace bot SDK display labels with the brain name before envelope output. |
| NapCat QQ | bigbang | Remove bot nickname semantics and use only the brain name for bot labels. |
| Dispatcher | bigbang | Require brain-owned name in dispatch context; remove adapter-name and generic fallbacks. |
| Pending/background delivery | bigbang | New output uses the current active profile name; old source-name fields remain provenance only. |
| Historical database rows | retained | Preserve existing rows exactly; canonicalize only newly built live reply context. |
| Tests and ICDs | bigbang | Rewrite the contract and expectations without legacy-shape coverage. |

Cutover enforcement:

- Rewrite all callers and callees in one commit.
- Reject the old registration request shape at the Pydantic boundary.
- A missing brain name is an operational contract failure, not a platform-name
  fallback condition.
- Any policy change requires a new user instruction.

## Target State

```text
platform account
  -> adapter supplies platform + callback + platform_bot_id
  -> brain registers transport and returns active character_name
  -> adapter stores last validated brain character_name
  -> platform bot mention/reply normalization uses that name
  -> /chat carries canonical bot labels and typed platform identity
  -> brain uses the same active profile for cognition and live reply fallback
```

Observable invariants:

- QQ id `3768713357` is still the typed QQ bot id.
- When the running brain profile name is `一之濑明日奈`, a QQ or Discord bot
  mention and bot reply display as `一之濑明日奈` even when the platform account
  nickname remains `杏山千纱`.
- Human platform labels retain their existing platform-specific policy.
- Heartbeat refresh updates adapter character-name state from the brain.
- Old conversation rows remain unchanged.
- New accepted-task and self-cognition assistant rows use the current active
  profile name.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Name authority | Use the active process-local brain profile. | It is the identity used by cognition and dialog in the same runtime. |
| Transport identity | Keep `platform_bot_id` adapter-owned and required. | The adapter alone discovers the native account id. |
| Synchronization | Return `character_name` on registration and heartbeat. | Both adapters already use this bounded control plane; no response-path call is added. |
| Adapter cache | Keep the last successfully validated brain name. | Heartbeat can refresh it without accepting platform or malformed values. |
| Normalizer input | Require `character_name` in Discord and QQ request-like inputs. | The boundary itself must prevent stale platform bot labels. |
| Reply history | Override only live reply context for rows authored by the platform bot id. | Current semantics become canonical without mutating history. |
| Background snapshots | Preserve snapshots as provenance but do not use them to name new output. | New visible output must follow current brain identity. |
| Debug adapter | Keep unchanged. | It neither fetches nor resolves a platform bot display name. |
| Failure | Fail registration/name validation closed. | A neutral or platform fallback would recreate multiple name authorities. |

## Contracts And Data Shapes

Registration and heartbeat request:

```python
{
    "platform": str,
    "callback_url": str,
    "platform_bot_id": str,
    "shared_secret": str,
    "timeout_seconds": float,
}
```

`display_name` is forbidden.

Registration and heartbeat response:

```python
{
    "status": str,
    "platform": str,
    "callback_url": str,
    "character_name": str,
}
```

`character_name` is required and non-empty after adapter validation.

Normalizer request-like objects for both QQ and Discord require:

```python
platform_bot_id: str
character_name: str
```

Bot mentions and bot replies use `character_name`; all other label sources
retain their existing contracts.

`DispatchContext` requires:

```python
source_platform_bot_id: str
source_character_name: str
```

No compatibility request field, alias attribute, adapter-side default, or
generic assistant-name fallback is allowed.

## LLM Call And Context Budget

- LLM call count before and after: unchanged.
- Response-path and background model routes: unchanged.
- Prompt schemas and context caps: unchanged.
- The only model-facing effect is correction of an existing readable bot
  mention or reply label before it enters the existing typed envelope path.

## Change Surface

### Create

- `src/adapters/runtime_registration.py`: shared strict extraction of the
  required brain-owned `character_name` from registration responses.
- `development_plans/active/short_term/brain_owned_adapter_character_name_bigbang_plan.md`:
  active work contract, later archived on completion.

### Modify: brain and dispatcher

- `src/kazusa_ai_chatbot/brain_service/contracts.py`: strict big-bang request
  and response schemas.
- `src/kazusa_ai_chatbot/brain_service/runtime_adapters.py`: pass required bot
  id into remote registration and return the required brain name.
- `src/kazusa_ai_chatbot/service.py`: supply the active name, remove remote
  display-name input, canonicalize bot reply context, and use the current name
  for accepted-task delivery.
- `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`: remove remote
  display-name state and require platform bot id.
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`: remove adapter and generic
  character-name fallbacks.
- `src/kazusa_ai_chatbot/dispatcher/task.py`: require the two brain-owned
  identity fields in `DispatchContext`.
- `src/kazusa_ai_chatbot/self_cognition/delivery.py`: use the current profile
  name for new dispatch.

### Modify: adapters

- `src/adapters/discord_adapter.py`: parse brain names and enforce them in bot
  mention/reply normalization.
- `src/adapters/napcat_qq_adapter/ws_adapter.py`: remove `bot_name`, parse
  brain names, and pass the canonical normalizer/reply inputs.
- `src/adapters/napcat_qq_adapter/mention_hydration.py`: use the brain name for
  bot-id mentions while leaving human cache behavior unchanged.
- `src/adapters/napcat_qq_adapter/reply_hydration.py`: use the brain name for a
  bot-id reply target.
- `src/adapters/napcat_qq_adapter/envelope_normalizer.py`: enforce the required
  brain name at the final adapter boundary.

### Modify: tests and ICDs

- `tests/test_runtime_adapter_registration.py`
- `tests/test_adapter_envelope_normalizers.py`
- `tests/test_service_background_consolidation.py`
- `tests/test_dispatcher_send_message_result.py`
- `tests/test_dispatcher_event_logging.py`
- `tests/test_background_work_delivery.py`
- `tests/test_delivery_mentions.py`
- `src/adapters/README.md`
- `src/adapters/napcat_qq_adapter/README.md`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/message_envelope/README.md`
- `src/kazusa_ai_chatbot/dispatcher/README.md`
- `development_plans/README.md`: add only this plan's lifecycle records while
  preserving all existing uncommitted hunks.

### Keep

- Keep `README.md`, `README_CN.md`, and `docs/HOWTO.md` unchanged by this plan.
- Keep `src/adapters/debug_adapter.py` unchanged.
- Keep message-envelope types, intake identity resolution, CQ projection,
  database repositories, profile loader, RAG, cognition, dialog,
  consolidation, and prompts unchanged.

## Overdesign Guardrail

- Actual problem: platform bot display names can disagree with the character
  name used by the brain.
- Minimal change: use the existing registration/heartbeat control plane to
  distribute one required brain name and enforce it at adapter and dispatch
  boundaries.
- Ownership boundaries: adapters own platform ids and syntax; the brain owns
  character name; deterministic code validates and maps; existing LLM stages
  consume the corrected semantic label.
- Rejected complexity: profile push notifications, per-message name fetches,
  database migration, feature flags, compatibility fields, alternate response
  shapes, alias services, retries beyond existing heartbeat, and prompt logic.
- Evidence threshold: a separately approved live-profile reload requirement
  or measured heartbeat propagation failure is required before adding another
  synchronization mechanism.

## Agent Autonomy Boundaries

- The parent agent may choose local expression and test-fixture mechanics only
  when they preserve the contracts above.
- The parent agent must not add architecture, compatibility layers, fallback
  paths, fields, features, or unrelated cleanup.
- Changes outside the named paths require a failing approved verification gate
  that cannot be resolved inside the existing surface.
- Search for existing equivalent behavior before creating a helper.
- Preserve surrounding style and line endings; avoid whole-file formatting.
- Review-only corrections are limited to named tests, ICDs, and this plan.
- If another agent modifies a target concurrently, stop editing that target,
  inspect both diffs, and continue only when both ownership sets can be
  preserved exactly.

## Implementation Order

1. Add focused deterministic contract assertions in the named test files.
2. Run the focused node ids and record failures caused by the old request,
   response, normalizer, reply, dispatcher, and background-name behavior.
3. Change the brain registration models and helper in one schema cutover.
4. Change Discord and NapCat registration consumers and normalizers.
5. Remove dispatcher adapter-name fallbacks and update current-name delivery
   callers.
6. Canonicalize historical bot reply context without writing history.
7. Run focused tests and loop only inside the approved surface until they pass.
8. Update subsystem ICDs to match the verified code contract.
9. Run syntax, static, focused, and broad deterministic regression gates.
10. Perform parent-only Independent Code Review, remediate in-scope findings,
    rerun affected gates, archive the plan, update the registry, and commit
    only scoped changes.

## Execution Model

- The user explicitly required no subagents and authorized single-agent
  fallback execution.
- The parent agent owns plan drafting, focused tests, production code, ICDs,
  verification, fresh-posture plan review, fresh-posture code review,
  lifecycle updates, selective staging, commit, and sign-off.
- Test contracts are established and run before production implementation.
- The code review gate is performed only after all planned verification.

## Progress Checklist

- [x] Stage 1 - focused failing contracts established.
  - Verify: named node ids fail for old registration/name behavior.
  - Evidence: the QQ and Discord normalizer node ids failed because bot
    mentions resolved to the stale platform label `杏山千纱` instead of the
    required brain name `一之濑明日奈`. A direct model-boundary probe confirmed
    the old registration schema accepted `display_name`, accepted a missing
    `platform_bot_id`, and accepted a response without `character_name`.
    Service-importing node ids were independently blocked at collection by
    another agent's concurrent removal of `_cognition_llm_config`; that file
    remains outside this plan and untouched.
  - Next: implement shared brain registration contract.
  - Sign-off: Codex parent, 2026-07-27.
- [x] Stage 2 - brain and adapter big-bang contract complete.
  - Verify: registration and normalizer focused tests pass.
  - Evidence: the strict request/response models, shared response validator,
    Discord consumer, NapCat consumer, mention hydration, reply hydration, and
    final normalizers were updated. The complete envelope-normalizer suite
    passed 23 tests and the complete runtime-adapter registration suite passed
    78 tests.
  - Next: dispatcher and background current-name paths.
  - Sign-off: Codex parent, 2026-07-27.
- [x] Stage 3 - dispatcher, reply, and background paths complete.
  - Verify: dispatcher, service reply, accepted-task, and self-cognition tests
    pass.
  - Evidence: `DispatchContext` now requires both identity fields; dispatcher
    adapter/generic fallbacks were removed; live bot reply context is
    canonicalized by bot id; accepted-task and self-cognition output use the
    active profile name. The combined focused suite passed 55 tests.
  - Next: ICD alignment and full verification.
  - Sign-off: Codex parent, 2026-07-27.
- [x] Stage 4 - ICD and regression verification complete.
  - Verify: every command below passes or an unrelated dirty-worktree failure
    is isolated and recorded without scope expansion.
  - Evidence: five subsystem ICDs now describe the verified big-bang
    ownership contract. Syntax, 156 focused tests, repository-wide non-live
    regression, static fallback scans, and scoped whitespace checks passed.
  - Next: independent code review.
  - Sign-off: Codex parent, 2026-07-27.
- [x] Stage 5 - parent-only Independent Code Review, lifecycle closeout, and
  commit complete.
  - Verify: diff ownership, style, contract, staged paths, and commit contents.
  - Evidence: all findings were remediated; 163 focused tests and the final
    repository-wide non-live gate passed; staged paths match the plan-owned
    allowlist; exact shared-registry hunks preserve concurrent work; the plan
    is archived in the same atomic change. The final commit hash is verified
    after commit and reported in the user handoff because a commit cannot
    embed its own stable hash.
  - Next: final user sign-off.
  - Sign-off: Codex parent, 2026-07-27.

- [ ] Stage 6 continue and close off the reminder steps of development_plans\active\short_term\cognition_core_v2_stage_llm_endpoint_routing_plan.md without subagent.

## Verification

### Syntax

- `venv\Scripts\python.exe -m py_compile src\adapters\runtime_registration.py src\adapters\discord_adapter.py src\adapters\napcat_qq_adapter\ws_adapter.py src\adapters\napcat_qq_adapter\mention_hydration.py src\adapters\napcat_qq_adapter\reply_hydration.py src\adapters\napcat_qq_adapter\envelope_normalizer.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\brain_service\runtime_adapters.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\dispatcher\remote_adapter.py src\kazusa_ai_chatbot\dispatcher\handlers.py src\kazusa_ai_chatbot\dispatcher\task.py src\kazusa_ai_chatbot\self_cognition\delivery.py tests\test_runtime_adapter_registration.py tests\test_adapter_envelope_normalizers.py tests\test_service_background_consolidation.py tests\test_dispatcher_send_message_result.py tests\test_dispatcher_event_logging.py tests\test_background_work_delivery.py tests\test_delivery_mentions.py`
  Expected: exit code 0.

### Focused deterministic tests

- `venv\Scripts\python.exe -m pytest tests\test_adapter_envelope_normalizers.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_runtime_adapter_registration.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_dispatcher_send_message_result.py tests\test_dispatcher_event_logging.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_background_work_delivery.py tests\test_delivery_mentions.py -q`

Expected: all pass as regular deterministic tests.

### Broader regression

- `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`

Expected: pass. Failures in pre-existing dirty files must be isolated by path
and reproduced without this plan's diff before classification; this plan must
not absorb unrelated fixes.

### Static contract scans

- `rg -n 'bot_name|Adapter-side display name fallback|adapter_text_attr\(adapter, "display_name"\)' src/adapters src/kazusa_ai_chatbot/dispatcher src/kazusa_ai_chatbot/service.py`
  - Expected: zero semantic bot-name fallback matches.
- `rg -n '"display_name"\s*:\s*(self\.bot_name|str\(user\.display_name\))' src/adapters`
  - Expected: zero registration or bot-identity matches.
- `rg -n "class RuntimeAdapterRegistrationRequest|class RuntimeAdapterRegistrationResponse|character_name" src/kazusa_ai_chatbot/brain_service src/adapters`
  - Expected: request/response and both adapter consumers visibly implement
    the one canonical field.
- `git diff --check -- src\adapters\runtime_registration.py src\adapters\discord_adapter.py src\adapters\napcat_qq_adapter src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\brain_service\runtime_adapters.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\self_cognition\delivery.py tests\test_runtime_adapter_registration.py tests\test_adapter_envelope_normalizers.py tests\test_service_background_consolidation.py tests\test_dispatcher_send_message_result.py tests\test_dispatcher_event_logging.py tests\test_background_work_delivery.py tests\test_delivery_mentions.py src\adapters\README.md src\kazusa_ai_chatbot\brain_service\README.md src\kazusa_ai_chatbot\message_envelope\README.md development_plans\active\short_term\brain_owned_adapter_character_name_bigbang_plan.md development_plans\README.md`
  - Expected: no whitespace errors in this plan's diff.

### Ownership and staging

- Compare final `git status --short` with the discovery snapshot.
- Inspect `git diff -- src\adapters\runtime_registration.py src\adapters\discord_adapter.py src\adapters\napcat_qq_adapter src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\brain_service\runtime_adapters.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\self_cognition\delivery.py tests\test_runtime_adapter_registration.py tests\test_adapter_envelope_normalizers.py tests\test_service_background_consolidation.py tests\test_dispatcher_send_message_result.py tests\test_dispatcher_event_logging.py tests\test_background_work_delivery.py tests\test_delivery_mentions.py src\adapters\README.md src\kazusa_ai_chatbot\brain_service\README.md src\kazusa_ai_chatbot\message_envelope\README.md development_plans\active\short_term\brain_owned_adapter_character_name_bigbang_plan.md development_plans\README.md`.
- Inspect `git diff --cached --name-only` and `git diff --cached`.
- Expected staged content: only this plan's source, tests, ICDs, archived plan,
  and exact registry lifecycle hunks.
- Expected unstaged content: every unrelated pre-existing user/agent change
  remains present and uncommitted by this plan.

## Independent Plan Review

Review mode: parent-only fresh-posture review, explicitly required by the user.

Surfaced issues and resolutions:

| Finding | Severity | Resolution |
| --- | --- | --- |
| Returning a brain name only at startup would leave adapters stale after a brain restart or active-profile change. | Blocker | Both registration and heartbeat return and validate `character_name`. |
| Leaving `display_name` optional in the request would preserve the old authority path. | Blocker | Remove it, require `platform_bot_id`, and forbid extra request fields. |
| Updating only event maps would leave direct normalizer and reply paths vulnerable. | Blocker | Require `character_name` in both normalizer inputs and override bot reply targets there. |
| NapCat direct reply metadata and `get_msg` can carry the old bot nickname. | Blocker | Pass brain name through reply hydration and enforce again in the final normalizer. |
| Dispatcher contains hidden adapter-name and `assistant` fallbacks. | Blocker | Require brain-owned dispatch identity fields and remove those fallbacks. |
| Pending accepted-task and self-cognition snapshots can carry an old brain name. | Blocker | Preserve snapshots as provenance while naming new output from the current active profile. |
| A reply to an old assistant row can recover an old stored display name. | Blocker | Override live reply display when the recovered author id equals `platform_bot_id`; do not mutate the row. |
| Debug has no registration/name resolution path. | Non-blocking | Keep debug unchanged and document it as pass-through/non-applicable. |
| Runtime profile reload semantics were undefined. | Blocker | Define authority as the running brain's startup-loaded active profile; live reload remains deferred. |
| Shared top-level docs and registry contain another agent's work. | Blocker | Keep top-level docs out of scope and stage only exact registry lifecycle hunks. |
| Normal execution guidance asks for subagents. | Blocker | Record the user's explicit no-subagent fallback authorization in Execution Model and both review gates. |
| The first verification draft used shell-unsafe regex quoting and path placeholders. | Blocker | Replace them with PowerShell-safe quoting and exact syntax/diff commands before approval. |

Review result: all blockers are resolved in this final plan. Status is
`approved` for execution.

## Independent Code Review

Run after all verification and before lifecycle completion or commit. Because
the user prohibited subagents, the parent must reread this plan, the prior
completed mention/identity plans, all changed source and tests, the complete
scoped diff, verification evidence, staged diff, and dirty-worktree snapshot
from a fresh-review posture.

Review:

- strict big-bang request/response consistency;
- absence of platform bot-name and generic assistant fallbacks;
- bot mention/reply canonicalization in QQ and Discord;
- current-name ownership for new dispatcher/background output;
- preservation of human label behavior and historical database rows;
- no new LLM call, prompt, DB migration, or per-message control-plane call;
- Python/CJK/test style;
- concurrent-change preservation and selective staging accuracy.

Fix only concrete findings inside the approved change surface. Record all
findings, fixes, rerun commands, residual risks, and approval state in
Execution Evidence.

Review result, 2026-07-27:

| Finding | Severity | Resolution |
| --- | --- | --- |
| Both request-like normalizers converted a non-string `character_name` with `str(...)`, so `None` could become the semantic label `None` instead of failing closed. | Blocker | Validate the raw value as a string before trimming in both normalizers; add QQ and Discord regression tests. |
| Both request-like normalizers tolerated an empty `platform_bot_id`, which could prevent the brain-name override from identifying a platform-bot mention. | Blocker | Require a non-empty string bot id in both normalizers; add QQ and Discord regression tests and align Discord event fixtures with the production SDK identity contract. |
| Brain registration, live reply canonicalization, accepted-task delivery, and self-cognition delivery could stringify a malformed profile name. | Blocker | Add one validated active-profile accessor in the service and explicit self-cognition validation; add registration and self-cognition regressions. |
| Discord heartbeat did not catch the transient `RuntimeError` raised while its account id is unavailable, unlike NapCat. | Blocker | Include `RuntimeError` in the bounded heartbeat failure path and add a loop-continuity regression. |
| Shared response validation, bot-id-scoped override behavior, human labels, persisted history, and background provenance matched the approved contracts. | Non-blocking | Retain the implemented design without additional surface. |

All blocker regressions failed before remediation and passed afterward. The
complete focused and broad deterministic suites passed after the fixes. No
unresolved code-review finding remains.

## Acceptance Criteria

This plan is complete when:

- The old registration request with `display_name` is rejected.
- Every runtime registration request requires `platform_bot_id`.
- Every successful registration/heartbeat response includes validated
  `character_name`.
- Discord and NapCat bot mention and reply labels use the brain name despite a
  conflicting platform name.
- NapCat has no `bot_name` semantic state or nickname fallback.
- Dispatcher and new background output have no adapter-name or generic
  assistant fallback.
- Replies to old assistant rows use the active brain name in live context
  without database mutation.
- Human label behavior remains unchanged.
- No conversation-history migration or LLM/prompt change exists.
- Focused and applicable broader tests pass.
- Parent-only independent review has no unresolved findings.
- The commit contains only plan-owned changes and leaves all other dirty
  worktree changes unstaged.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Adapter receives malformed or old brain response | Fail startup or heartbeat validation without accepting a platform name | Registration parser tests |
| Rename is not visible until synchronization | Registration plus periodic heartbeat refreshes the adapter cache | Heartbeat rename tests |
| Historical row exposes old reply label | Canonicalize current reply context by platform bot id | Service reply fallback tests |
| Human names are accidentally overwritten | Override only ids equal to `platform_bot_id` | Mixed bot/human normalizer tests |
| Background output uses a stale source snapshot | Resolve new output name from active profile at execution | Accepted-task and self-cognition tests |
| Concurrent agent work enters this commit | Hash/diff checks and selective hunk staging | Final staged-diff review |

## Execution Evidence

- Discovery snapshot: target production, adapter, dispatcher, ICD, and focused
  test files were clean; `development_plans/README.md` and unrelated
  cognition/control-console/top-level files were already modified.
- Plan review: parent-only review completed; all findings listed in
  `Independent Plan Review` were addressed before approval.
- Stage 1 red evidence:
  - `test_qq_normalizer_rewrites_cq_mentions_as_readable_tokens` failed with
    `@杏山千纱` where `@一之濑明日奈` was required.
  - `test_discord_normalizer_rewrites_tags_as_readable_tokens` failed with
    `@杏山千纱` where `@一之濑明日奈` was required.
  - The direct registration-model probe printed
    `legacy_display_name_accepted='old'`,
    `missing_platform_bot_id_accepted=''`, and a response model dump without
    `character_name`.
  - Service-dependent collection was blocked by the unrelated concurrent
    `_cognition_llm_config` import inconsistency and is deferred to the normal
    focused verification gate after that worktree state settles.
- Stage 2 implementation evidence:
  - `venv\Scripts\python.exe -m pytest
    tests\test_adapter_envelope_normalizers.py -q`: 23 passed.
  - `venv\Scripts\python.exe -m pytest
    tests\test_runtime_adapter_registration.py -q`: 78 passed.
  - Immediate `py_compile` checks passed for every changed adapter,
    registration, brain-contract, remote-adapter, service, and CJK-bearing
    test file in this stage.
- Stage 3 implementation evidence:
  - `venv\Scripts\python.exe -m pytest
    tests\test_service_background_consolidation.py
    tests\test_dispatcher_send_message_result.py
    tests\test_dispatcher_event_logging.py
    tests\test_background_work_delivery.py
    tests\test_delivery_mentions.py -q`: 55 passed.
  - Immediate `py_compile` checks passed for dispatcher contracts and
    handlers, self-cognition delivery, and the service reply/background
    changes.
  - No database write or migration command was run; historical rows and
    persisted source-name snapshots were left intact.
- Stage 4 verification evidence:
  - The exact planned `py_compile` gate passed for all changed production and
    focused test modules.
  - The combined focused deterministic command passed 156 tests.
  - `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`
    completed with exit code 0 in 251.3 seconds.
  - Both forbidden fallback scans returned zero matches; the positive contract
    scan showed the strict schemas, shared parser, and both adapter consumers.
  - `git diff --check` completed with no whitespace errors; Git emitted only
    the repository's normal CRLF conversion notices.
- Stage 5 parent-only code-review evidence:
  - The parent reread this plan, both prior completed adapter identity plans,
    every changed source/test diff, verification evidence, and the live dirty
    worktree from a fresh-review posture.
  - Seven added review regressions first failed on non-string identity
    conversion, missing bot identity, and Discord heartbeat termination, then
    passed after the scoped corrections.
  - The complete focused command passed 163 tests after review remediation.
  - The repository-wide non-live command passed with exit code 0 in 249.8
    seconds after the first review remediation and again in 243.9 seconds
    after the final bot-id boundary correction.
  - Exact syntax, forbidden-fallback, positive-contract, and scoped whitespace
    gates passed again.
  - The staged implementation diff contains only the plan-owned source, tests,
    subsystem ICDs, shared adapter helper, archived execution record, and exact
    registry lifecycle hunks. Every unrelated pre-existing worktree change
    remains unstaged.
  - Residual operational risk is bounded to synchronization timing: a renamed
    startup-loaded profile becomes visible after adapter registration or the
    next successful heartbeat. Live profile reload remains explicitly
    deferred.
