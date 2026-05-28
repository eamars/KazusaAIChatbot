# decontextualizer scope users referent bugfix plan

## Summary

- Goal: replace the full-history referent retry direction with a one-pass
  scoped-user identity roster so the decontextualizer can resolve person
  references that are visible in reply/history context without reading full
  message history.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `test-style-and-execution`, `debug-llm`, `py-style`, `cjk-safety`
- Overall cutover strategy: bigbang replacement of the full-history retry path
  with a compatible first-pass prompt payload extension.
- Highest-risk areas: prompt candidate-list bias, local LLM over-resolving
  genuinely missing `他/她` references, scope roster overgrowth in busy group
  chats, accidental RAG ownership drift, and prompt-rule inconsistency from
  appending instructions instead of rewriting the decontextualizer contract.
- Acceptance criteria: the decontextualizer first pass receives a bounded
  neutral `scope_users` identity roster; the full-history retry path is removed
  from production behavior; the original QQ-style failure resolves `他` to
  `蚝爹油` in one live LLM call; no-anchor negative probes keep `他`
  unresolved; deterministic state/payload tests and focused regressions pass.

## Context

On 2026-05-27 the QQ group input:

```text
@杏山千纱 还不报警抓他吗？
```

failed decontextualization:

```text
Decontextualizer output: output="@杏山千纱 还不报警抓他吗？"
referents=[{"phrase": "他", "referent_role": "object", "status": "unresolved"}]
RAG2 skipped output: reason="缺少以下指代对象: 他"
```

The DB export showed the reply/history context already contained the needed
person name, `蚝爹油`. The failure was not a RAG blind spot and not primarily
missing history. RAG is downstream and must not resolve pronouns. The failure
was a local LLM attention and identity-linking problem inside the
decontextualizer: the model saw a name in visible reply/history context but did
not bind that identity to `他`.

A superseded local retry attempt explored one full-history retry after
unresolved first-pass output. The new POC evidence supports a better next
stage: provide a neutral scoped-user roster in the first pass instead of
increasing the history text window or adding a second LLM call. The cleanup
stage removes the retry attempt's dirty code/test diff and its untracked plan
artifact if present.

POC evidence copied into this plan: current baseline failed; prompt-only POC
failed; `scope_users` roster-only POC resolved `他` to `蚝爹油`; role-annotated
POC also resolved but is more bias-prone. Two no-anchor negative probes kept
`他` unresolved across baseline, roster-only POC, and role-annotated POC.

This plan turns the POC result into production work while avoiding candidate
labels, evidence roles, deterministic likely-referent selection, and RAG
changes.

## Mandatory Skills

- `development-plan`: load before editing, executing, reviewing, lifecycle
  changes, or sign-off.
- `local-llm-architecture`: load before changing prompt inputs, state shape,
  response-path call count, graph orchestration, RAG boundaries, or LLM context budget.
- `test-style-and-execution`: load before adding, changing, or running deterministic,
  patched-LLM, or live LLM tests.
- `debug-llm`: load before live local LLM checks and readable quality reports.
- `py-style`: load before editing Python source or tests.
- `cjk-safety`: load before editing Python files containing CJK prompt or test strings.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Check `git status --short` before editing.
- Read this entire plan after any automatic context compaction and before
  continuing implementation, verification, lifecycle updates, or final
  reporting.
- Read this entire plan after signing off any major progress checklist stage
  and before starting the next stage.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Do not read `.env`.
- Before adding scope-users implementation code, remove the dirty full-history
  retry attempt from the working tree using path-limited cleanup. Remove the
  superseded untracked retry-plan artifact and its registry row if present.
  Preserve the active scope-users plan and registry row. Do not revert
  unrelated user changes.
- Do not modify RAG planner, RAG specialists, retrieval tools, Cache2,
  conversation-progress storage, memory, scheduler, adapters, persistence, or
  dialog generation for this plan.
- Do not make RAG resolve pronouns or referents.
- Do not increase `CONVERSATION_HISTORY_LIMIT`,
  `CHAT_HISTORY_RECENT_LIMIT`, or the decontextualizer `chat_history` text
  window.
- Do not fetch additional DB rows. Build `scope_users` only from data already
  present in the live turn state: current user, active character, current
  prompt context mentions/addressing, reply context identity fields, and the
  already-loaded channel history rows.
- Do not include message body text inside `scope_users`.
- Do not include deterministic `likely_referent`, ranking scores,
  `evidence_roles`, gender labels, or reason strings in production
  `scope_users`.
- Do not add a new LLM evaluator, helper agent, summarizer, candidate-ranker,
  feature flag, compatibility mode, or background repair path.
- Prompt changes must rewrite the relevant decontextualizer contract sections
  so evidence order, field descriptions, pronoun rules, and output rules remain
  coherent. Do not append a disconnected `scope_users` section.
- `scope_users` is an identity table, not a candidate-answer list. The prompt
  must require a bridge from `user_input`, `reply_context`,
  `prompt_message_context`, or `chat_history` before using a scoped user as a
  resolved referent.
- Real LLM tests must run one case at a time with `-s`; inspect each emitted
  trace before running the next live LLM case.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Remove the superseded dirty full-history retry attempt before adding
  scope-users implementation code. Record the cleanup in `Execution Evidence`.
- Add a production `scope_users` state/payload contract for the
  decontextualizer first pass.
- Build the roster deterministically from already-loaded, prompt-safe turn
  context and deduplicate by stable identity.
- Include the active character, current speaker, prompt mentions/addressed
  users, reply identity when present, and visible channel-history speakers.
- Keep roster rows compact: display name, platform user id, global user id, and
  aliases only. Use an empty alias list until a trusted alias source is already
  available in the current state.
- Pass `scope_users` to `call_msg_decontexualizer` when non-empty.
- Rewrite the decontextualizer prompt contract to define `scope_users` as a
  neutral identity table and to preserve unresolved behavior when no text
  bridge exists.
- Remove the full-history unresolved-referent retry path from production
  behavior, including retry-only schema, retry orchestration, retry prompt
  rules, and retry acceptance helper/tests.
- Preserve RAG skip semantics for all-unresolved referents.
- Preserve the normal decontextualizer output shape:

```python
{
    "decontexualized_input": str,
    "referents": list[dict[str, str]],
}
```

- Add deterministic tests for roster construction, payload inclusion, prompt
  contract text, and absence of full-history retry.
- Add live LLM tests proving the original QQ-style failure resolves with
  `scope_users` and no-anchor negative cases remain unresolved.
- Remove executable POC artifacts after production live LLM validation exists.
- Run every verification command listed in this plan.

## Deferred

- Do not add evidence-role annotations to production `scope_users`.
- Do not add a deterministic referent shaper, likely-candidate selector, gender
  inference, relationship scorer, or recency scorer.
- Do not introduce a second LLM call, evaluator loop, or full-history fallback
  for this failure mode.
- Do not change RAG skip behavior when all referents are unresolved.
- Do not change cognition clarification wording except through existing
  downstream behavior after `referents` changes.
- Do not add DB migrations, persistent metrics collections, operator endpoints,
  or new adapter contracts.
- Do not keep executable POC code after production implementation and live LLM
  validation replace it.

## Cutover Policy

Overall strategy: bigbang replacement of the retry behavior, compatible payload
extension for the first pass.

| Area | Policy | Instruction |
|---|---|---|
| First-pass decontextualizer payload | compatible | Add optional `scope_users` to the existing first-pass payload. Existing callers without `scope_users` still work. |
| Dirty retry attempt in the working tree | bigbang | Remove the superseded retry attempt before starting scope-users implementation. Preserve only the new plan and registry row. |
| Full-history retry path | bigbang | Remove retry orchestration and retry-only prompt/schema/helper behavior. Do not keep it as a fallback. |
| RAG boundary | compatible | Preserve existing RAG behavior. RAG continues to skip all-unresolved referents and does not resolve pronouns. |
| Prompt contract | bigbang | Rewrite the relevant decontextualizer prompt sections coherently. Do not append an isolated rule. |
| Tests | bigbang | Replace retry-focused tests with scope-roster tests and live LLM evidence. |
| POC artifacts | bigbang | Remove executable POC files and transient POC traces after production live LLM validation is recorded. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For `bigbang` areas, implement the planned behavior directly instead of
  adding feature flags, compatibility aliases, alternate retry paths, or dual
  behavior.
- For `compatible` areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

`persona_supervisor2` builds a bounded scoped-user identity roster before the
persona graph starts. The first decontextualizer call receives:

```python
{
    "chat_history_recent": recent_channel_history_for_decontextualizer,
    "scope_users": scope_users,
}
```

`chat_history_recent` remains capped to `CHAT_HISTORY_RECENT_LIMIT`. The
decontextualizer never receives full channel message text solely because a
referent was unresolved.

When the prompt context already bridges a pronoun to a person name, alias,
reply author, mentioned user, or history speaker, the LLM may use
`scope_users` to output the stable display name. When no bridge exists, the LLM
must keep the phrase unresolved.

The downstream state remains:

```python
{
    "decontexualized_input": str,
    "referents": list[ReferentResolution],
}
```

No downstream consumer reads `scope_users`; it is a decontextualizer-only
prompt input.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix owner | Decontextualizer first-pass input owns the fix | Pronoun resolution belongs before RAG and cognition. |
| Roster semantics | `scope_users` is an identity table, not candidate answers | This gives the local LLM names/ids without forcing a selection. |
| History text | Keep recent bounded history text | The POC solved the failure without full-history text expansion. |
| Retry path | Remove full-history retry from production behavior | The user prefers one pass and POC evidence supports one pass. |
| Evidence roles | Do not ship role annotations | Roster-only solved the failure with less candidate bias. |
| Alias field | Include `aliases: []` initially | Keeps the contract stable without inventing an alias source. |
| RAG behavior | Preserve all-unresolved RAG skip | RAG is not the referent resolver. |

## Contracts And Data Shapes

Add this prompt-facing shape:

```python
class ScopeUser(TypedDict):
    display_name: str
    platform_user_id: str
    global_user_id: str
    aliases: list[str]
```

Add to `GlobalPersonaState`:

```python
scope_users: NotRequired[list[ScopeUser]]
```

Human payload shape sent to the decontextualizer when non-empty:

```json
{
  "scope_users": [
    {
      "display_name": "蚝爹油",
      "platform_user_id": "673225019",
      "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
      "aliases": []
    }
  ]
}
```

Roster construction rules:

- Deduplicate by `global_user_id` when present, else by `platform_user_id`,
  else by non-empty `display_name`.
- Prefer the latest non-empty display name seen in the already-loaded context.
- Include the active character with `display_name=character_profile["name"]`,
  `platform_user_id=platform_bot_id`, and
  `global_user_id=character_profile["global_user_id"]`.
- Include the current speaker from `platform_user_id`, `global_user_id`, and
  `user_name`.
- Include prompt mentions/addressed users from `prompt_message_context` when
  their ids or display names are present.
- Include reply identity only from typed `reply_context` identity fields when
  present; do not parse arbitrary `reply_excerpt` text deterministically.
- Include speakers from the already-loaded channel history rows.
- Do not include message text, timestamps, roles, scores, or evidence labels in
  roster rows.

Remove these retry-only contracts from production:

- `ReferentRetryContext`
- `GlobalPersonaState.referent_retry_context`
- `REFERENT_RETRY_INSTRUCTION`
- `retry_resolves_unresolved_referents(...)`
- retry-specific prompt payload and prompt rules

## LLM Call And Context Budget

Affected response-path call:

| Call | Before this plan | After this plan |
|---|---|---|
| `MSG_DECONTEXTUALIZER_LLM` | One first pass plus a possible full-history retry when all referents are unresolved in the current working tree | Exactly one first pass |

Context inputs after this plan:

- unchanged `user_input`, `prompt_message_context`, capped `chat_history`,
  `reply_context`, `channel_topic`, and `indirect_speech_context`;
- new compact `scope_users` roster;
- removed retry-only full-history text and `referent_retry_context`.

Budget policy:

- No new response-path LLM call is added.
- The text history window remains capped by `CHAT_HISTORY_RECENT_LIMIT`.
- The roster must be bounded to already-loaded scope users and must not include
  message body text.
- If exact tokenization is unavailable during implementation, measure payload
  size by serialized JSON character count for baseline versus changed payload
  in deterministic tests or debug output.

## Change Surface

### Delete

- Dirty full-history retry attempt changes from the working tree before
  implementation starts:
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - `src/kazusa_ai_chatbot/nodes/referent_resolution.py`
  - `tests/test_msg_decontexualizer.py`
  - `tests/test_persona_supervisor2.py`
  - `tests/test_referent_resolution.py`
  - `tests/test_persona_supervisor2_decontext_retry_live_llm.py`
  - `development_plans/archive/completed/bugfix/decontextualizer_unresolved_referent_retry_bugfix_plan.md`
  - the `decontextualizer_unresolved_referent_retry_bugfix_plan.md` registry
    row in `development_plans/README.md`
- Retry-only logic in `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`.
- Retry-only schema in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`.
- Retry-only helper in `src/kazusa_ai_chatbot/nodes/referent_resolution.py`
  when no production caller remains.
- Retry-only deterministic/live tests after replacement coverage exists.
- Executable POC artifacts after production live LLM tests replace them:
  - `experiments/persona_supervisor2_msg_decontexualizer_scope_users_baseline.py`
  - `experiments/persona_supervisor2_msg_decontexualizer_scope_users_poc.py`
  - `experiments/decontext_scope_users_live_poc_runner.py`
  - `test_artifacts/decontext_scope_users_live_poc/`

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Build `scope_users` from the current already-loaded turn context.
  - Pass `scope_users` in `initial_persona_state`.
  - Keep decontextualizer `chat_history_recent` on the recent channel-history
    projection.
  - Remove the full-history retry branch.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add `ScopeUser`.
  - Add `GlobalPersonaState.scope_users`.
  - Remove retry-only schema once callers are gone.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Rewrite the prompt contract to explain `scope_users`.
  - Include `scope_users` in the human payload when non-empty.
  - Remove retry-only prompt rules and payload handling.
- `tests/test_persona_supervisor2.py`
  - Replace retry orchestration tests with roster construction and one-pass
    handoff tests.
- `tests/test_msg_decontexualizer.py`
  - Add payload and prompt-contract coverage for `scope_users`.
  - Remove retry-payload tests.
- `tests/test_referent_resolution.py`
  - Remove retry-helper tests if the helper is removed.
- `tests/test_persona_supervisor2_decontext_scope_users_live_llm.py`
  - Create live LLM tests for the original failure and negative no-anchor
    probes.

### Create

- No new production module is required unless implementation shows the roster
  builder would make `persona_supervisor2.py` materially harder to review. If a
  new module is necessary, stop and update this plan before creating it.

### Keep

- RAG skip logic in `stage_1_research`.
- Existing `referents` output contract.
- Existing typed message envelope and prompt message context contracts.
- Human-authored POC conclusions copied into this plan. POC source files and
  transient trace directories are removed after production live tests exist.

## Overdesign Guardrail

- Actual problem: the local decontextualizer fails to bind a third-person
  pronoun to a person already visible in reply/history context.
- Minimal change: add a compact first-pass scoped-user identity roster and
  update the prompt contract so the LLM can use names already bridged by the
  text.
- Ownership boundaries: deterministic code builds and bounds the roster; the
  LLM makes semantic referent judgments; RAG retrieves evidence only after
  decontextualization; cognition decides stance and clarification behavior.
- Rejected complexity: full-history retry, evaluator loops, candidate ranking,
  likely-referent labels, evidence roles, gender inference, aliases from new DB
  lookups, feature flags, and RAG-side pronoun resolution.
- Evidence threshold: add rejected complexity only after live LLM traces show
  roster-only first pass cannot resolve a required class of visible-context
  references or creates unacceptable false unresolved/false resolved behavior
  that cannot be fixed by prompt-contract wording.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside `persona_supervisor2`,
  `persona_supervisor2_schema`, `persona_supervisor2_msg_decontexualizer`,
  `referent_resolution`, and the listed tests as high-scrutiny changes that
  require a plan update before implementation.
- The responsible agent must search for existing equivalent roster, mention,
  and prompt-projection helpers before adding new helper functions.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites outside the decontextualizer, or broad
  refactors.
- If the plan and code disagree, the responsible agent must preserve the
  plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

0. Parent removes the superseded dirty retry attempt before implementation.
   - Run `git status --short` and record the dirty files.
   - Verify the dirty production/test files listed in `Change Surface/Delete`
     correspond to the previous full-history retry attempt.
   - Remove those path-limited changes, the untracked retry live test, the
     untracked retry-plan artifact, and the retry-plan registry row.
   - Preserve this plan and the active scope-users registry row in
     `development_plans/README.md`.
   - Run `git status --short` again and record the cleanup result.
1. Parent adds deterministic schema/payload tests.
   - Add `tests/test_msg_decontexualizer.py` coverage proving
     `scope_users` appears in the human payload when present and no
     retry-only payload field appears.
   - Run the focused test and record the expected failure before production
     implementation.
2. Parent adds deterministic persona orchestration tests.
   - Add `tests/test_persona_supervisor2.py` coverage proving
     `scope_users` includes active character, current speaker, visible history
     speakers, and mention/reply identities; decontextualizer is called once
     for all-unresolved first-pass output; no full-history retry state is sent.
   - Run the focused tests and record the expected failure before production
     implementation.
3. Parent starts one production-code subagent.
   - Provide this plan, mandatory skills, focused test failures, and the
     approved change surface.
   - The subagent edits production code only and closes after planned
     production changes are complete.
4. Production-code subagent updates `persona_supervisor2_schema.py`.
   - Add `ScopeUser` and `scope_users`.
   - Remove retry-only schema after production callers are removed.
5. Production-code subagent updates `persona_supervisor2.py`.
   - Add deterministic roster construction inside the approved boundary.
   - Pass `scope_users` to the initial persona state.
   - Remove full-history retry orchestration.
6. Production-code subagent updates
   `persona_supervisor2_msg_decontexualizer.py`.
   - Rewrite the prompt contract coherently.
   - Include `scope_users` in payload when present.
   - Remove retry-only payload handling and prompt rules.
7. Production-code subagent removes retry-only helper code in
   `referent_resolution.py` when no callers remain.
8. Parent updates/removes deterministic tests to match the final contract.
   - Remove retry-only tests that no longer represent supported behavior.
   - Keep RAG skip tests for all-unresolved referents.
9. Parent runs focused deterministic verification.
10. Parent adds live LLM tests and raw trace artifacts.
    - Add one original-failure test requiring `他` to resolve to `蚝爹油`.
    - Add two negative no-anchor tests requiring `他` to remain unresolved.
11. Parent runs each live LLM test one at a time with `-s` and inspects each
    trace before continuing.
12. Parent removes executable POC artifacts.
    - Remove the copied decontextualizer POC files, runner, and transient POC
      trace directory listed in `Change Surface/Delete`.
    - Keep production live LLM traces and execution evidence.
13. Parent runs broader regression verification.
14. Parent starts one independent code-review subagent.
15. Parent remediates review findings that are inside this plan's change
    surface and reruns affected verification.
16. Parent records execution evidence and updates lifecycle status only after
    review approval and verification pass.

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
  live LLM trace preparation, and validation work while the production-code
  subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused deterministic test contract established
  - Covers: implementation steps 0-2.
  - Verify: previous dirty retry attempt is removed from the worktree before
    new tests are added; focused new tests are present and fail for the missing
    `scope_users` production behavior.
  - Evidence: record cleanup status, commands, and failure snippets in
    `Execution Evidence`.
  - Handoff: production-code subagent starts at Stage 2.
  - Sign-off: `Codex/2026-05-28` after evidence is recorded.
- [x] Stage 2 - production scope roster contract implemented
  - Covers: implementation steps 3-7.
  - Verify: `venv\Scripts\python.exe -m py_compile` on changed production
    Python files.
  - Evidence: record changed files, removed retry surfaces, and compile output.
  - Handoff: parent resumes at Stage 3.
  - Sign-off: `Hooke/Codex/2026-05-28` after production subagent reports completion.
- [x] Stage 3 - deterministic tests updated and passing
  - Covers: implementation steps 8-9.
  - Verify: focused deterministic test commands in `Verification`.
  - Evidence: record passing output.
  - Handoff: continue to live LLM validation.
  - Sign-off: `Codex/2026-05-28` after verification passes.
- [x] Stage 4 - live LLM validation complete
  - Covers: implementation steps 10-11.
  - Verify: run each live LLM test one at a time and inspect trace artifacts.
  - Evidence: record trace paths and quality judgment.
  - Handoff: remove POC artifacts.
  - Sign-off: `Codex/2026-05-28` after each live trace is inspected.
- [x] Stage 5 - POC artifacts removed
  - Covers: implementation step 12.
  - Verify: executable POC files and transient POC trace directory listed in
    `Change Surface/Delete` are absent.
  - Evidence: record path checks and retained production evidence paths.
  - Handoff: continue to broader regression.
  - Sign-off: `Codex/2026-05-28` after cleanup is recorded.
- [x] Stage 6 - regression verification complete
  - Covers: implementation step 13.
  - Verify: broader commands in `Verification`.
  - Evidence: record passing output or unrelated failures with classification.
  - Handoff: continue to independent code review.
  - Sign-off: `Codex/2026-05-28` after verification is recorded.
- [x] Stage 7 - independent code review complete
  - Covers: implementation steps 14-15.
  - Verify: review subagent findings are resolved or explicitly accepted as
    residual risk.
  - Evidence: record findings, fixes, rerun commands, and approval status.
  - Handoff: final lifecycle/sign-off.
  - Sign-off: `Zeno/Codex/2026-05-28` after review approval.
- [x] Stage 8 - lifecycle update and final sign-off complete
  - Covers: implementation step 16.
  - Verify: `git status --short` and registry/status updates reflect the final
    state.
  - Evidence: record final changed files and commands.
  - Handoff: plan may move to `archive/completed/bugfix/`.
  - Sign-off: `Codex/2026-05-28` after completion criteria are met.

## Verification

Working-tree cleanup gate before implementation:

```powershell
git status --short
git diff -- `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py `
  src\kazusa_ai_chatbot\nodes\referent_resolution.py `
  tests\test_msg_decontexualizer.py `
  tests\test_persona_supervisor2.py `
  tests\test_referent_resolution.py
```

Abort cleanup if the diff contains unrelated user work. If the diff is only
the superseded retry attempt, remove it with path-limited commands:

```powershell
git restore --worktree -- `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py `
  src\kazusa_ai_chatbot\nodes\referent_resolution.py `
  tests\test_msg_decontexualizer.py `
  tests\test_persona_supervisor2.py `
  tests\test_referent_resolution.py
if (Test-Path -LiteralPath 'tests\test_persona_supervisor2_decontext_retry_live_llm.py') {
  Remove-Item -LiteralPath 'tests\test_persona_supervisor2_decontext_retry_live_llm.py'
}
if (Test-Path -LiteralPath 'development_plans\archive\completed\bugfix\decontextualizer_unresolved_referent_retry_bugfix_plan.md') {
  Remove-Item -LiteralPath 'development_plans\archive\completed\bugfix\decontextualizer_unresolved_referent_retry_bugfix_plan.md'
}
# Use apply_patch to remove only the retry-plan registry row from
# development_plans\README.md. Preserve the active
# decontextualizer_scope_users_referent_bugfix_plan.md row, then verify:
Select-String -LiteralPath 'development_plans\README.md' -Pattern 'decontextualizer_unresolved_referent_retry_bugfix_plan'
git status --short
```

Expected cleanup result: the `Select-String` command prints no retry-plan
registry row, while the scope-users active bugfix registry row remains.

Static checks:

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py `
  src\kazusa_ai_chatbot\nodes\referent_resolution.py
```

Focused deterministic tests:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_msg_decontexualizer.py `
  tests\test_persona_supervisor2.py `
  tests\test_referent_resolution.py -q
```

Live LLM tests must run one by one:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_resolves_original_qq_failure `
  -q -s
```

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_keeps_absent_person_unresolved `
  -q -s
```

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_keeps_gender_name_probe_unresolved `
  -q -s
```

Broader regression:

```powershell
venv\Scripts\python.exe -m pytest `
  tests\test_persona_supervisor2.py `
  tests\test_msg_decontexualizer.py `
  tests\test_persona_supervisor2_rag2_integration.py `
  tests\test_cognition_clarification_consumers.py `
  tests\test_referent_resolution.py -q
```

Manual greps:

```powershell
rg -n "referent_retry_context|REFERENT_RETRY_INSTRUCTION|retry_resolves_unresolved_referents" src tests
```

Expected grep result after implementation: no production or active test
references remain. Historical archive records and experiment artifacts may
still contain these strings.

POC removal check after live LLM validation:

```powershell
Remove-Item -LiteralPath `
  'experiments\persona_supervisor2_msg_decontexualizer_scope_users_baseline.py', `
  'experiments\persona_supervisor2_msg_decontexualizer_scope_users_poc.py', `
  'experiments\decontext_scope_users_live_poc_runner.py' `
  -ErrorAction SilentlyContinue
$repoRoot = (Resolve-Path -LiteralPath '.').Path
$pocTracePath = Resolve-Path -LiteralPath 'test_artifacts\decontext_scope_users_live_poc' -ErrorAction SilentlyContinue
if ($pocTracePath -and $pocTracePath.Path.StartsWith($repoRoot)) {
  Remove-Item -LiteralPath $pocTracePath.Path -Recurse
}
Test-Path -LiteralPath 'experiments\persona_supervisor2_msg_decontexualizer_scope_users_baseline.py'
Test-Path -LiteralPath 'experiments\persona_supervisor2_msg_decontexualizer_scope_users_poc.py'
Test-Path -LiteralPath 'experiments\decontext_scope_users_live_poc_runner.py'
Test-Path -LiteralPath 'test_artifacts\decontext_scope_users_live_poc'
```

Expected result after cleanup: all four commands print `False`.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread the POC reports, this plan, and relevant source/test context
from a fresh-review posture.

Review scope: POC evidence is carried forward accurately; RAG does not gain
pronoun-resolution ownership; dependencies, decisions, status, registry rows,
and artifacts are present; schema, orchestration, prompt, tests, verification,
and evidence instructions are concrete; no optional fallbacks, evaluator loops,
candidate rankers, or broad refactors remain; the one-pass scope-users design
fits local LLM latency and context-budget constraints.

Record blockers, non-blocking findings, required edits, residual risks, and
approval status in `Execution Evidence`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope: project rules and style compliance; ownership boundaries,
fallback/shim risk, prompt/RAG payload leaks, persistence risk, brittle
fixtures, and blast radius; alignment with `Must Do`, `Deferred`, autonomy
boundaries, change surface, contracts, order, verification, and acceptance
criteria; regression and handoff quality, including live traces, execution
evidence, cleanup, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The superseded dirty full-history retry attempt has been removed before
  scope-users implementation work starts, and the cleanup is recorded in
  `Execution Evidence`.
- `scope_users` is present in the first-pass decontextualizer payload whenever
  the roster is non-empty.
- `scope_users` contains only neutral identity rows and no message text,
  evidence roles, likely-candidate labels, scores, gender labels, or reason
  strings.
- The decontextualizer prompt coherently explains `scope_users` as an identity
  table and preserves unresolved behavior when no text bridge exists.
- Full-history unresolved-referent retry behavior is removed from production
  code and active tests.
- RAG still skips all-unresolved referents and does not resolve pronouns.
- Deterministic tests prove payload shape, one-pass orchestration, roster
  construction, and retry removal.
- Live LLM tests prove the original QQ-style failure resolves to `蚝爹油` and
  the no-anchor negative probes remain unresolved.
- Executable POC files and transient POC traces are removed after production
  tests and live LLM traces replace them.
- Verification commands pass or any unrelated failures are identified and
  explicitly recorded.
- Independent code review is complete and approved.
- Execution evidence records commands, outputs, live trace paths, changed
  files, review findings, and residual risks.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Roster biases the model to pick a visible user when `他` is truly unresolved | Prompt frames roster as identity table and requires a text bridge; no evidence roles or likely-candidate labels | Live no-anchor and gender-name negative tests |
| Removing retry loses a useful fallback for rare cases | POC showed roster solves the observed failure in one pass; rejected complexity can be revisited only with new evidence | Original-failure live LLM test and regression trace review |
| Busy group roster grows too large | Build only from already-loaded scope, deduplicate, omit message text, and measure serialized payload size | Deterministic roster test and payload-size evidence |
| Prompt edit becomes contradictory | Rewrite evidence-order and field sections coherently instead of appending | Prompt-contract assertions and independent code review |
| RAG boundary drifts | Keep RAG code unchanged except existing skip behavior | Focused RAG/referent regression tests |

## Execution Evidence

- 2026-05-28 plan review fixed cleanup scope, POC-removal scope, retry-artifact
  dependency, and line-budget issues; draft remains pending execution approval.
- 2026-05-28 cleanup executed by Codex after explicit user instruction. The
  inspected dirty production/test diff contained only the superseded
  full-history retry attempt.
- Cleanup commands: path-limited `git restore --worktree --` for the seven
  dirty production/test files; `Remove-Item -LiteralPath` for the untracked
  retry live test and untracked retry-plan artifact; `apply_patch` removed only
  the retry-plan registry row from `development_plans\README.md`.
- Cleanup verification: restored production/test paths have no remaining diff;
  both removed retry artifacts return `Test-Path=False`; the active
  scope-users registry row remains and the retry-plan row is absent.
- Stage 1 remains incomplete: focused scope-users deterministic tests have not
  been added or run yet.
- 2026-05-28 user approved the plan and requested subagent execution; status
  moved to `in_progress`.
- 2026-05-28 Stage 1 deterministic contract added by parent in
  `tests/test_msg_decontexualizer.py` and `tests/test_persona_supervisor2.py`.
  Focused baseline command:
  `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py::test_decontexualizer_forwards_scope_users_as_neutral_identity_table tests\test_persona_supervisor2.py::test_persona_supervisor2_builds_scope_users_for_first_pass_only -q`.
  Expected result before production implementation: 2 failures. Failure
  snippets: decontextualizer payload assertion raised `KeyError:
  'scope_users'`; persona handoff assertion raised `KeyError: 'scope_users'`.
- 2026-05-28 Stage 2 implemented by production subagent Hooke. Changed
  production files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`, and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`.
  `src/kazusa_ai_chatbot/nodes/referent_resolution.py` was unchanged because
  no active retry-only production leftovers were present.
- Stage 2 verification: parent reran
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\nodes\referent_resolution.py`;
  command exited 0 with no stdout.
- Stage 2 focused contract rerun:
  `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py::test_decontexualizer_forwards_scope_users_as_neutral_identity_table tests\test_persona_supervisor2.py::test_persona_supervisor2_builds_scope_users_for_first_pass_only -q`;
  result: 2 passed.
- Stage 3 focused deterministic verification:
  `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2.py tests\test_referent_resolution.py -q`;
  result: 28 passed.
- Stage 4 live LLM validation ran one case at a time with `-m live_llm -q -s`.
  Initial `original_qq_failure` run failed because the newly authored test
  fixture used the wrong older “解决掉” context and no reply excerpt; the trace
  was inspected and excluded from acceptance evidence.
- Stage 4 corrected original-failure live command:
  `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_resolves_original_qq_failure -m live_llm -q -s`;
  result: passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_scope_users_live_llm__original_qq_failure__20260528T095823461111Z.json`.
- Stage 4 negative live command:
  `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_keeps_absent_person_unresolved -m live_llm -q -s`;
  result: passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_scope_users_live_llm__absent_person_unresolved.json`.
- Stage 4 gender-name negative live command:
  `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_keeps_gender_name_probe_unresolved -m live_llm -q -s`;
  result: passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_scope_users_live_llm__gender_name_probe_unresolved.json`.
- Stage 4 human-readable review artifact:
  `test_artifacts/diagnostics/decontext_scope_users_production_live_review_20260528.md`.
  Quality judgment: original QQ-style failure resolved `他` to `蚝爹油`;
  both no-anchor probes kept `他` unresolved.
- Stage 5 POC cleanup removed executable POC files and transient traces:
  `experiments/persona_supervisor2_msg_decontexualizer_scope_users_baseline.py`,
  `experiments/persona_supervisor2_msg_decontexualizer_scope_users_poc.py`,
  `experiments/decontext_scope_users_live_poc_runner.py`, and
  `test_artifacts/decontext_scope_users_live_poc/`.
  Path checks returned `False` for all four paths. Human-authored diagnostic
  reports and production live traces were retained.
- Stage 6 broader regression verification:
  `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2.py tests\test_msg_decontexualizer.py tests\test_persona_supervisor2_rag2_integration.py tests\test_cognition_clarification_consumers.py tests\test_referent_resolution.py -q`;
  result after final test edits: 48 passed.
- Stage 6 manual retry-surface grep:
  `rg -n "referent_retry_context|REFERENT_RETRY_INSTRUCTION|retry_resolves_unresolved_referents" src tests`;
  result: no matches, exit code 1.
- Parent local review then tightened roster ordering so chronological history
  fills the roster before current-turn identities override older display
  names. Rerun verification: production/test `py_compile` exited 0;
  `tests/test_persona_supervisor2.py::test_persona_supervisor2_builds_scope_users_for_first_pass_only`
  passed; broader regression rerun result: 48 passed; retry-surface grep
  remained no matches.
- Stage 7 independent code review completed by subagent Zeno. Findings:
  duplicate helper should reuse `utils.text_or_empty`; roster helper docstrings
  needed Args/Returns; one test fixture dict had over-indented fields. All
  concrete findings were fixed inside the approved change surface.
- Stage 7 remediation verification: `py_compile` for changed production files
  and `tests/test_persona_supervisor2.py` exited 0; focused roster/payload
  tests passed; `git diff --check` exited 0; broader regression rerun result:
  48 passed.
- Stage 7 accepted residual risks: live evidence covers one configured local
  model and three cases; ambiguous multi-person, alias-only, and duplicate
  display-name cases remain future validation candidates; current live traces
  do not capture the exact model name; `scope_users` remains present in the
  internal persona graph snapshot, so future generic state consumers must avoid
  prompt/persistence leakage.
- Stage 8 lifecycle completion: plan status set to `completed`; registry is
  updated to remove the active bugfix row and add the completed bugfix archive
  row; final `git status --short` records the changed source/test/plan files
  and the unrelated pre-existing
  `development_plans/active/short_term/rag_agent_package_reorganization_plan.md`.
