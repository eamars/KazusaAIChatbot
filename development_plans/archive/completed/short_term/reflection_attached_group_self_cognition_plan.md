# reflection attached group self cognition plan

## Summary

- Goal: Build group-channel self-cognition review from the existing reflection
  cycle path, with source-aligned group delivery and no private fallback.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `database-data-pull` for Stage 0 diagnostics, `py-style`,
  `test-style-and-execution`, and `cjk-safety` when editing Python files that
  contain CJK prompt text.
- Overall cutover strategy: bigbang for self-cognition target binding,
  compatible for reflection-cycle hourly and daily reflection records.
- Highest-risk areas: mixing reflection output into live cognition, creating a
  second self-cognition scheduler, speaking in the wrong group channel,
  over-triggering from noisy group traffic, and changing existing reflection
  promotion semantics.
- Acceptance criteria: implementation is complete only when accepted Stage 0
  manual experiment evidence remains recorded, group review runs from the
  reflection cycle cadence without new self-cognition config, eligible group
  scopes are limited to groups where the character spoke in the last 24 hours,
  group source cases bind to the same group channel, existing hourly and daily
  reflection behavior remains intact, all verification gates pass,
  independent code review approves the result, and post-sign-off experiment
  cleanup is recorded.

## Context

Current reflection already owns a 15-minute background cadence. The default
`REFLECTION_WORKER_INTERVAL_SECONDS` is `900`, and the reflection worker runs
hourly-slot work on that cadence. Reflection scope selection already uses a
monitor rule equivalent to "latest assistant message inside monitor window" in
`conversation_history`, and the monitor window is 24 hours.

Current self-cognition source collection is not sufficient for group review.
Production collection currently includes scheduled future cognition and active
commitments, while documented trigger flags include group chat review. Current
self-cognition target binding prefers a known private channel and only falls
back to the source channel when private is unavailable. That behavior is not
approved for this work.

The target architecture is shared observation, not shared semantic output:

```text
conversation_history
  -> reflection-cycle 15-minute activity window
  -> reflection hourly/daily consumers
  -> group self-cognition review consumer
```

Reflection output remains audit and promotion evidence. Raw reflection output
must not become a direct instruction to self-cognition or dialog.

## Mandatory Skills

- `development-plan-writing`: preserve this work contract, lifecycle status,
  Stage 0 gate, and execution evidence.
- `local-llm-architecture`: keep semantic judgment in existing cognition
  stages, keep operational routing and delivery decisions deterministic, and
  avoid adding retry loops or prompt fields for adapter feasibility.
- `database-data-pull`: use read-only live database diagnostics for Stage 0;
  do not read `.env`.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files with CJK prompt text.

## Mandatory Rules

- This plan is approved after Stage 0 evidence was collected and manually
  accepted on 2026-05-18. Production changes may start from this approved
  plan only.
- Stage 0 is an experiment under `experiments/`; it must not mutate
  production collections or call platform adapters. The base projection probe
  should be read-only and zero-LLM; a manual decision probe may call the
  existing self-cognition cognition path when explicitly requested, but must
  still disable consolidation, delivery, scheduler mutation, memory mutation,
  and adapter sends.
- Do not add a dedicated self-cognition interval, group-review interval,
  group-review enable flag, adapter-capability config, retry config, or
  fallback config.
- Group review scheduling must attach to the existing reflection cycle cadence.
- The group lookup criterion must match the reflection monitor rule: group
  scopes are eligible only when the character has an assistant message in that
  group inside the last 24 hours.
- Group source self-cognition cases must bind to the same group channel.
- Do not use private-channel lookup for group-source self-cognition cases.
- Do not add retry or fallback behavior for group delivery in this plan.
- Do not feed raw hourly or daily reflection output into normal cognition.
  Reflection output can contribute only through existing promoted and gated
  reflection context, or as non-actionable source references when explicitly
  approved by this plan.
- The LLM must receive semantic labels and bounded visible context, not raw
  telemetry that requires it to infer activity thresholds.
- If implementation touches LLM prompt text, prompt templates, cognition
  source-packet rendering, graph state fields visible to prompts, or prompt
  schemas, the agent must rewrite the affected prompt or prompt-facing source
  packet contract under `local-llm-architecture`; do not patch in one-off
  wording. The rewrite must audit the smallest semantic question, keep stable
  contract material in the system prompt, keep current-run source data in the
  human/source packet, include explicit generation procedure plus input/output
  format for structured prompt stages, avoid development-plan or migration
  language in runtime prompts, avoid hard-coded concrete character names, and
  run a prompt-render check in addition to syntax and unit tests.
- Deterministic code owns scope selection, windowing, target binding,
  cooldown, limits, idempotency, and delivery execution.
- LLM stages own only the semantic decision of whether an eligible source case
  merits silence, audit, progress maintenance, or visible speech.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.

## Must Do

- Add a Stage 0 read-only experiment under
  `experiments/reflection_attached_group_self_cognition_probe/`.
- Stage 0 must output manually reviewable group activity windows and proposed
  self-cognition case shells before any production implementation starts.
- Reuse the reflection monitor rule for eligible group scopes.
- Add a reflection-attached group review path that does not introduce new
  self-cognition scheduling config.
- Preserve existing hourly-slot, daily-channel, and daily-global reflection
  behavior.
- Add a group self-cognition case source for active group review.
- Bind group self-cognition delivery target to the source group channel.
- Remove or bypass private fallback for group-source self-cognition target
  binding.
- Add deterministic semantic labels for group activity windows before
  cognition sees them.
- Add idempotency and cooldown behavior so one noisy group window cannot
  repeatedly trigger visible speech.
- Add focused deterministic tests for Stage 0 projection helpers, reflection
  attachment, source-aligned binding, fallback removal, and existing reflection
  regressions.
- Record execution evidence and run independent code review before completion.

## Deferred

- Do not add adapter channel capability reporting.
- Do not add delivery retry.
- Do not add private fallback for group sources.
- Do not add a new self-cognition worker interval or source-specific env flag.
- Do not redesign daily reflection promotion.
- Do not change global character growth.
- Do not migrate or rewrite historical reflection runs.
- Do not mutate historical conversation rows or self-cognition attempts.
- Do not implement private-channel 30-minute review in this plan.
- Do not add multi-group batch speech or group broadcast policy beyond the
  source-aligned group review case.

## Stage 0 Gate

Stage 0 was mandatory for approval and is now accepted for this plan. It is
not production implementation.

Stage 0 must create a read-only experiment under:

```text
experiments/reflection_attached_group_self_cognition_probe/
```

The planned experiment entrypoint is:

```text
experiments/reflection_attached_group_self_cognition_probe/probe_group_activity_windows.py
```

The experiment must collect live database evidence using existing project DB
settings and must write output under an ignored experiment output directory.
The base projection probe must not call LLMs, adapters, dispatcher sends,
scheduler mutation, memory mutation, or self-cognition delivery. The manual
decision probe may call the existing self-cognition cognition path only to
inspect the decision for one selected 15-minute group window; it must not
send, consolidate, mutate scheduler state, mutate memory, or execute delivery.

The Stage 0 output must include:

- selected group scopes admitted by the 24-hour assistant-message monitor
  rule;
- excluded group scopes and the reason they were excluded;
- 15-minute activity windows for admitted group scopes;
- bounded visible-context rows for each candidate window;
- source message references sufficient for audit;
- semantic labels for activity level, participant diversity, bot recency,
  direct-address signal, noise level, and response risk;
- a proposed self-cognition case shell for each candidate window;
- a proposed same-group `delivery_target` shell;
- a compact hourly aggregation preview showing how four 15-minute windows can
  still feed hourly reflection.

Manual validation must confirm all of these before Stage 1 starts:

- At least three group scopes are admitted by the 24-hour assistant-message
  rule, or at least six 15-minute group windows exist across the inspected
  period if live traffic is lower.
- The output contains at least one candidate window that appears worth
  cognition review and at least one window that should remain silent/audit-only.
- Every proposed group case targets the same group channel as its source.
- No private-channel target appears in any group-source case.
- The semantic labels are understandable without reading raw metrics.
- The visible context is enough for a human reviewer to judge whether the case
  should enter self-cognition.
- The hourly aggregation preview is compatible with existing hourly reflection
  concepts and does not require changing daily reflection semantics.

Stage 0 evidence must remain in `Execution Evidence` with artifact paths,
command output, reviewer notes, and the manual decision. The owner accepted
the Stage 0 evidence on 2026-05-18 based on the twelve-window real-LLM probe
and fabricated-mood probe. Excluded-scope emission and hourly aggregation
preview remain required implementation evidence, but they no longer block
plan approval.

## Cutover Policy

Overall strategy: bigbang for self-cognition group target binding, compatible
for reflection-cycle reflection records.

| Area | Policy | Instruction |
|---|---|---|
| Group source target binding | bigbang | Group-source self-cognition binds to the same group channel. Do not preserve private fallback. |
| Private source target binding | bigbang | Private-source self-cognition binds to the same private source channel. Do not search for an alternate channel. |
| Internal or source-less cases | bigbang | Cases without a concrete source channel fail target binding unless an existing non-group source already carries an explicit concrete target. |
| Reflection cadence | compatible | Reuse `REFLECTION_WORKER_INTERVAL_SECONDS`; do not add self-cognition group cadence config. |
| Reflection hourly and daily runs | compatible | Preserve existing run kinds, prompt contracts, daily synthesis, and promotion behavior. |
| Stage 0 experiment | compatible | Add read-only experiment files under `experiments/`; they are not production runtime code. |
| Historical data | compatible | Do not migrate or rewrite historical conversation, reflection, scheduled-event, or self-cognition attempt rows. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- If an area is `bigbang`, replace the old behavior directly instead of
  preserving fallback or compatibility paths.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to this policy requires user approval before implementation.

## Data Migration

No data migration is approved.

- Do not backfill reflection runs.
- Do not backfill self-cognition attempts.
- Do not update historical conversation rows.
- Do not update scheduled events.

## Agent Autonomy Boundaries

- The agent may choose local helper names only after preserving the public
  ownership boundaries in this plan.
- The agent must not add new config, fallback paths, retries, compatibility
  shims, adapter contracts, broad abstractions, or unrelated cleanup.
- The agent must stop and report a blocker if the reflection cycle cannot
  safely host the group review sidecar without changing reflection promotion
  semantics.
- The agent may add small projection helpers for 15-minute group activity
  windows when tests define their input and output shapes first.
- The agent must keep production behavior changes inside reflection-cycle
  scheduling, self-cognition source collection, target binding, and focused
  tests/docs unless this plan is revised.

## Target State

The reflection cycle remains the background timing owner. On each reflection
tick, the system can derive bounded 15-minute group activity windows from the
same monitored-channel source used by reflection. Reflection continues to
persist hourly slots and daily channel syntheses. The group self-cognition
consumer receives only selected windows with semantic labels and bounded
visible context.

A group self-cognition case has this target invariant:

```json
{
  "trigger_kind": "group_chat_trigger_review",
  "target_scope": {
    "platform": "qq",
    "platform_channel_id": "source-group-id",
    "channel_type": "group",
    "user_id": null
  },
  "delivery_target": {
    "platform": "qq",
    "platform_channel_id": "source-group-id",
    "channel_type": "group",
    "source_kind": "self_cognition_source_channel",
    "fallback_reason": ""
  }
}
```

The exact production shape may include existing required fields, but it must
not include a private target for a group source.

## Design Decisions

- Use the existing reflection monitor rule for group eligibility: the
  character must have an assistant message in the group in the last 24 hours.
- Attach group self-cognition review to the reflection cycle cadence instead
  of adding self-cognition group scheduling config.
- Keep reflection and self-cognition as separate consumers of shared activity
  windows.
- Keep hourly and daily reflection prompt contracts unchanged unless Stage 0
  evidence proves that a small projection helper is required.
- Derive semantic labels deterministically before invoking cognition.
- Allow self-cognition to select silence, audit, progress maintenance, or
  visible speech; do not force speech.
- Bind visible speech to the source group only.

## LLM Call And Context Budget

- Stage 0 base projection must use zero LLM calls.
- Stage 0 manual decision probe may use one existing self-cognition cognition
  call for one selected window when explicitly requested.
- Activity-window collection must use zero LLM calls.
- Existing hourly reflection keeps its current reflection LLM budget.
- Existing daily reflection keeps its current daily synthesis LLM budget.
- One group self-cognition case may use the existing self-cognition budget:
  at most one cognition call and, only if visible speech is selected, one
  dialog render call.
- Group review must not add RAG calls unless a later approved plan adds a
  bounded topic follow-up source.
- Group review source packets must stay within the existing
  `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT`.

## Change Surface

Approved Stage 0 surface:

- `experiments/reflection_attached_group_self_cognition_probe/**`

Expected production surface after Stage 0 approval:

- `src/kazusa_ai_chatbot/reflection_cycle/selector.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
- `src/kazusa_ai_chatbot/self_cognition/models.py`
- `src/kazusa_ai_chatbot/self_cognition/tracking.py` only if idempotency
  source identity needs a group-review source kind
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- focused tests under `tests/`

Out of scope unless this plan is revised:

- adapter implementations;
- dispatcher delivery protocol;
- scheduler task execution;
- global character growth;
- memory promotion;
- normal live-chat graph behavior;
- daily reflection prompt schema;
- `.env` or deployment secrets.

## Overdesign Guardrail

The actual problem is that group self-cognition lacks a sufficient production
source and the current target fallback can send to the wrong channel. The
minimal change is to reuse reflection-cycle monitored group activity as the
source, create same-group self-cognition cases from bounded 15-minute windows,
and remove private fallback for group-source cases.

Rejected complexity:

- no new self-cognition interval config;
- no group-review feature flag;
- no adapter capability reporting;
- no delivery retry;
- no fallback channel;
- no raw reflection output as action input;
- no new daily reflection schema;
- no broad reflection-worker rewrite.

Evidence required before adding flexibility:

- Stage 0 manual output proves that the shared 15-minute window lacks required
  fields;
- deterministic tests prove an existing contract cannot represent the case;
- user approves the additional field, module, config, or behavior.

## Implementation Order

1. Stage 0 read-only experiment.
   - Add experiment files under
     `experiments/reflection_attached_group_self_cognition_probe/`.
   - Run the experiment against live DB using project DB settings.
   - Save output artifacts under the experiment output directory.
   - Manually inspect the output against the Stage 0 gate.
   - Record evidence in this plan.
   - Acceptance state: complete as of 2026-05-18. If a rerun contradicts the
     accepted evidence, stop and update this plan before production changes.
2. Add focused tests for activity-window projection.
   - Test admitted group scopes, excluded groups, semantic labels, and bounded
     visible context.
   - Run the focused tests and record the expected pre-implementation failure.
3. Implement the activity-window projection helper.
   - Keep it deterministic and read-only.
   - Reuse reflection monitor inputs.
   - Run focused tests.
4. Add focused tests for source-aligned self-cognition target binding.
   - Prove group source binds to same group.
   - Prove private fallback is not used for group source.
   - Prove missing source target fails closed.
5. Implement target-binding cutover.
   - Remove or bypass private fallback according to this plan.
   - Run target-binding tests.
6. Add focused tests for group self-cognition case collection.
   - Prove semantic labels and visible context enter the case.
   - Prove idempotency/cooldown suppresses repeated noisy windows.
7. Implement group review case collection.
   - Keep LLM-visible packet bounded and semantic.
   - Run self-cognition source tests.
8. Attach group review to the reflection cycle.
   - Use reflection cadence and monitored windows.
   - Do not add new config.
   - Preserve hourly/daily reflection behavior.
9. Run integration and regression tests.
   - Include reflection worker tests and self-cognition worker/source tests.
10. Update READMEs and this plan's execution evidence.
11. Run independent code review and remediate approved findings.
12. Record final implementation sign-off after verification and independent
    code review pass.
13. After final sign-off, remove every artifact this plan added under
    `experiments/`, including
    `experiments/reflection_attached_group_self_cognition_probe/**` and the
    experiment-specific unignore rule in `experiments/.gitignore`. Keep the
    Stage 0 learning and artifact paths in this plan; do not keep the
    experiment source as a long-term repo file after sign-off.

## Progress Checklist

- [x] Stage 0 - read-only experiment evidence accepted
  - Covers: implementation order step 1.
  - Verify: experiment command output, artifact paths, and manual reviewer
    notes are recorded in `Execution Evidence`.
  - Evidence required: accepted Stage 0 manual validation.
  - Handoff: next agent starts at Stage 1 from this approved plan.
  - Sign-off: `Codex/2026-05-18`.
- [x] Stage 1 - activity-window tests and helper complete
  - Covers: implementation order steps 2-3.
  - Verify: focused activity-window tests pass.
  - Evidence required: expected failing test before implementation and passing
    output after implementation.
  - Handoff: Stage 2 completed in this pass.
  - Sign-off: `Codex/2026-05-18`.
- [x] Stage 2 - source-aligned target binding complete
  - Covers: implementation order steps 4-5.
  - Verify: self-cognition delivery-target tests pass.
  - Evidence required: tests prove no private fallback for group sources.
  - Handoff: Stage 3 completed in this pass.
  - Sign-off: `Codex/2026-05-18`.
- [x] Stage 3 - group review source collection complete
  - Covers: implementation order steps 6-7.
  - Verify: self-cognition source tests pass.
  - Evidence required: tests prove bounded semantic case creation and duplicate
    suppression.
  - Handoff: Stage 4 completed in this pass.
  - Sign-off: `Codex/2026-05-18`.
- [x] Stage 4 - reflection-cycle attachment complete
  - Covers: implementation order step 8.
  - Verify: reflection worker tests and self-cognition integration tests pass.
  - Evidence required: tests prove no new config and preserved reflection
    hourly/daily behavior.
  - Handoff: Stage 5 completed in this pass.
  - Sign-off: `Codex/2026-05-18`.
- [x] Stage 5 - docs, regression, and independent code review complete
  - Covers: implementation order steps 9-12.
  - Verify: all commands in `Verification` pass, docs are updated, and
    independent code review is recorded.
  - Evidence required: command outputs, review findings, fixes, and residual
    risks.
  - Handoff: Stage 6 cleanup completed after implementation sign-off.
  - Sign-off: `Codex/2026-05-18`.
- [x] Stage 6 - post-sign-off experiment cleanup complete
  - Covers: implementation order step 13.
  - Verify: `git status --short -- experiments` shows no planned tracked
    experiment probe files, and `experiments/.gitignore` no longer contains
    the experiment-specific unignore rule.
  - Evidence required: cleanup diff/status and confirmation that Stage 0
    learning remains recorded in this plan.
  - Handoff: plan completed and ready for archive.
  - Sign-off: `Codex/2026-05-18`.

## Verification

Stage 0 verification:

```powershell
venv\Scripts\python.exe experiments\reflection_attached_group_self_cognition_probe\probe_group_activity_windows.py
```

The command must be read-only and must write artifacts under the experiment
output directory.

Focused verification after implementation must include:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_delivery_target.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_integration.py -q
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_activity_windows.py -q
venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py -q
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_stage1c_service.py -q
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_prompt_contracts.py -q
```

If implementation touches any LLM prompt, prompt-facing source-packet text, or
prompt schema, verification must also include evidence that the affected prompt
or source-packet contract was rewritten under the prompt-writing rule, plus a
focused prompt-render check for the edited prompt path using representative
group-review payloads. Record the exact command and rendered-contract result
in `Execution Evidence`.

Post-sign-off cleanup verification must include:

```powershell
git status --short -- experiments
rg -n "reflection_attached_group_self_cognition_probe" experiments\.gitignore
```

After cleanup, the first command must show no planned tracked experiment probe
files. The second command must return no match; exit code 1 is acceptable for
`rg` when no match is found.

## Independent Plan Review

### 2026-05-18 approval review

- Reviewer mode: active agent fresh-review posture; no separate reviewer was
  available in this session.
- Inputs reviewed: `development_plans/README.md`, `plan_contract.md`,
  `execution_gates.md`, `cutover_policy.md`, `local-llm-architecture`, this
  plan, and the recorded Stage 0 experiment evidence.
- Blockers: none after the edits in this approval pass.
- Resolved review findings: stale pre-approval blocker language was replaced
  with accepted Stage 0 status, the owner-approved Stage 0 learning was
  recorded, prompt-writing rules were copied into mandatory rules for any LLM
  touch, and post-sign-off cleanup was added for everything introduced under
  `experiments/`.
- Non-blocking residual risks: the observed action-spec/route mismatch is
  recorded as a learning but is explicitly out of scope for this plan; the
  production group-review source must avoid biased legacy source wording and
  use active group-review semantics; excluded-scope and hourly aggregation
  evidence must still be produced during implementation verification.
- Approval status: approved for implementation under this plan.

## Independent Code Review

Before marking this plan completed, run an independent code review gate.

The review must check:

- plan alignment;
- no new self-cognition schedule config;
- no adapter capability contract;
- no group-source private fallback;
- no delivery retry;
- reflection hourly/daily behavior preserved;
- raw reflection output not fed directly into cognition;
- semantic labels are deterministic and test-covered;
- Stage 0 evidence matches the implemented source behavior;
- test coverage and verification evidence are current.

Findings that require new config, new fallback, adapter contract changes, or
reflection prompt schema changes require user approval before implementation.

## Acceptance Criteria

- Stage 0 experiment evidence was manually accepted on 2026-05-18 and remains
  recorded.
- The plan and registry are updated to `approved` before Stage 1 starts.
- Group review is driven by the reflection cycle cadence.
- No new self-cognition group-review config exists.
- Eligible groups are limited to groups where the character spoke in the last
  24 hours according to the reflection monitor rule.
- Group-source self-cognition cases target the same group channel.
- Private fallback is not used for group-source self-cognition.
- Existing hourly reflection, daily channel reflection, and daily global
  promotion behavior remain compatible.
- If LLM prompt, prompt-facing source-packet, or prompt schema text is touched,
  the affected prompt or source-packet contract is rewritten under the
  prompt-writing rule, and prompt-render evidence is recorded.
- All verification commands pass.
- Independent code review approves the implementation.
- After final implementation sign-off, every artifact this plan added under
  `experiments/` is removed, including the experiment-specific
  `experiments/.gitignore` unignore rule.

## Risks

- The reflection monitor rule may include assistant rows that were persisted
  before a failed delivery receipt. Stage 0 must inspect whether this appears
  in live data before approval.
- The shared 15-minute windows may be too noisy for self-cognition. Stage 0
  must prove semantic labels can separate review-worthy windows from noise.
- Reflection and self-cognition can contend for LLM capacity. The production
  implementation must keep group review bounded and must not add unbounded
  per-channel LLM calls.
- Removing private fallback changes current self-cognition delivery behavior.
  The cutover is intentional, but tests must make the changed behavior
  explicit.
- Stage 0 found that existing route classification can be too permissive when
  a generated `speak` action says to remain silent. This is not fixed in this
  plan by owner instruction; record it for a separate self-cognition routing
  quality plan if needed.
- Stage 0 source wording can bias the LLM if it resembles the existing
  self-check/private trigger source. Production group review must use active
  group-review semantics and bounded group-window evidence.

## Execution Evidence

### 2026-05-18 Stage 0 Manual Decision Probe

- Scope: one monitor-eligible group selected from the last three hours using
  the 24-hour assistant-message monitor rule.
- Experiment file:
  `experiments/reflection_attached_group_self_cognition_probe/probe_group_activity_windows.py`
- Command:
  `venv\Scripts\python.exe experiments\reflection_attached_group_self_cognition_probe\probe_group_activity_windows.py`
- Result: exit code 0.
- Selected source: `qq` group `491307527`, `217` messages in the three-hour
  lookback, latest assistant evidence had delivery status `delivered`.
- Selected 15-minute window:
  `2026-05-18T06:00:00+00:00` to `2026-05-18T06:15:00+00:00`,
  `1` message.
- Deterministic labels: `quiet`, `one_speaker`, `not_in_window`,
  `ambient_group_context`, `recent`, `medium`, `unclear`.
- Target binding evidence: `target_scope.channel_type` is `group`,
  `target_scope.platform_channel_id` is `491307527`,
  `delivery_target.channel_type` is `group`,
  `delivery_target.platform_channel_id` is `491307527`,
  `fallback_reason` is empty.
- Self-cognition decision: selected route `audit_only`, output mode `silent`,
  no action candidate, no action specs, logical stance `DIVERGE`, character
  intent `DISMISS`.
- Route effect: `production_write` is `false`; consumer is `audit_log`.
- Artifact directory:
  `experiments/reflection_attached_group_self_cognition_probe/output/2026-05-18T064706.738306_0000_scope_0b8b6d29f462_2026-05-18T060000+0000/`
- Verification:
  `venv\Scripts\python.exe -m py_compile experiments\reflection_attached_group_self_cognition_probe\probe_group_activity_windows.py`
  exited 0.
- Manual gate status at time of run: partial evidence only. This run validates
  one selected group decision path and same-group target shell; later
  twelve-window and fabricated-mood probes superseded this as the approval
  basis.

### 2026-05-18 Stage 0 Twelve-Window Manual Decision Probe

- Scope: one active monitor-eligible group selected by message volume from the
  last three completed hours, while retaining the 24-hour assistant-message
  monitor rule for eligibility.
- Experiment file:
  `experiments/reflection_attached_group_self_cognition_probe/probe_group_activity_windows.py`
- Command:
  `venv\Scripts\python.exe experiments\reflection_attached_group_self_cognition_probe\probe_group_activity_windows.py`
- Result: exit code 0 after 12 real self-cognition cognition calls.
- Summary artifact:
  `experiments/reflection_attached_group_self_cognition_probe/output/2026-05-18T065636.588541_0000_scope_0b8b6d29f462_2026-05-18T034500+0000_2026-05-18T064500+0000/experiment_summary.json`
- Evaluation range:
  `2026-05-18T03:45:00+00:00` to `2026-05-18T06:45:00+00:00`,
  12 fixed 15-minute windows.
- Selected source: `qq` group `491307527`, `217` messages in the three-hour
  lookback, latest assistant evidence had delivery status `delivered`.
- Verification: artifact contains 12 full `self_cognition_input` payloads and
  12 full `self_cognition_output` payloads; every target stayed bound to group
  `491307527`; no private targets appeared; route effects recorded zero
  production writes.
- Result pattern: all 12 windows selected `audit_only` with output mode
  `silent`; all 12 produced no action specs and no action candidate.
- Window message counts: `0`, `25`, `125`, `63`, `1`, `2`, `0`, `0`, `0`,
  `1`, `0`, `0`.
- Manual gate status at time of run: stronger cognition evidence collected.
  Later owner acceptance treated this run, together with the fabricated-mood
  probe, as sufficient Stage 0 evidence for plan approval. Excluded scopes and
  hourly aggregation preview remain implementation verification requirements.

### 2026-05-18 Fabricated Mood Variant Probe

- Scope: same experiment path, one active monitor-eligible group with empty
  15-minute windows skipped.
- Fabricated `current_mood` values:
  `正向：心情明亮，带着轻快的好奇和参与意愿`,
  `中性：情绪平稳，先观察事实，不急着靠近或回避`,
  `负向：心情低沉，带着疲惫和轻微抗拒，不想贸然介入`.
- Command:
  `venv\Scripts\python.exe experiments\reflection_attached_group_self_cognition_probe\probe_group_activity_windows.py`
- Result: exit code 0 after 15 real self-cognition cognition calls.
- Summary artifact:
  `experiments/reflection_attached_group_self_cognition_probe/output/2026-05-18T072624.217708_0000_scope_0b8b6d29f462_2026-05-18T041500+0000_2026-05-18T071500+0000/experiment_summary.json`
- Verification: artifact contains 15 full `self_cognition_input` payloads and
  15 full `self_cognition_output` payloads; every target stayed bound to group
  `491307527`; no private targets appeared; route effects recorded zero
  production writes.
- Empty-window handling: 5 non-empty windows processed, 7 empty windows
  skipped.
- Mood sensitivity result: in the first high-activity window, `正向` selected
  `action_candidate` with one visible `speak` spec, `中性` also selected
  `action_candidate` but the action spec text said to keep silent, and `负向`
  selected `audit_only` with no action specs.
- Risk finding: action-spec presence alone is too permissive for route
  classification; a `speak` action whose surface requirement says "keep
  silent" can still become an `action_candidate`.

### 2026-05-18 Stage 0 Learnings Accepted For Plan Approval

- Owner decision: Stage 0 results are acceptable for plan approval; do not fix
  the observed self-cognition route/action-spec issue in this plan.
- The selected active group path produced full real-LLM inputs and outputs for
  twelve 15-minute windows, with same-group target shells and no private
  target leakage.
- Empty 15-minute windows added no useful cognition signal in the fabricated
  mood run and should be skipped before self-cognition.
- `current_mood` is visible to self-cognition and can affect the decision when
  the group window has strong evidence. The positive, neutral, and negative
  Chinese mood strings diverged only in the high-activity window.
- Existing source wording can bias the cognition input toward a system
  self-check/no-response interpretation. The production source packet must use
  explicit active group-review wording, semantic group-window labels, and
  bounded visible context.
- The route/action-spec mismatch is real evidence but is out of scope for this
  plan by owner instruction. The implementation must not opportunistically fix
  that issue while adding the reflection-attached source.
- The selected cadence remains the existing reflection cadence: 15 minutes for
  group review. Private 30-minute review remains deferred.

### 2026-05-18 Implementation Evidence

- Execution mode: main agent owned focused tests and review; production code
  was delegated to one worker subagent and then reviewed by the main agent.
- TDD red evidence before production implementation:
  `venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_activity_windows.py -q`
  failed because `reflection_cycle.activity_windows` did not exist;
  `venv\Scripts\python.exe -m pytest tests/test_self_cognition_delivery_target.py -q`
  failed because resolver still used private-channel fallback;
  `venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py -q`
  failed because group-review source collection and source-packet contract did
  not exist; and
  `venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_stage1c_worker.py::test_worker_tick_runs_group_review_on_reflection_cadence -q`
  failed because the reflection worker did not call group review.
- Implemented activity-window projection in
  `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`: non-empty
  15-minute group windows, bounded visible context, semantic labels, and
  hourly aggregation preview.
- Implemented source-aligned target binding in
  `src/kazusa_ai_chatbot/self_cognition/sources.py`: valid group and private
  source cases bind to the same source channel, and missing or invalid concrete
  sources fail with `missing_delivery_target`.
- Implemented group review source cases in
  `src/kazusa_ai_chatbot/self_cognition/sources.py` and
  `src/kazusa_ai_chatbot/self_cognition/projection.py`: group cases are built
  from reflection-selected group activity windows, empty windows are skipped,
  source packets use active group-review wording, and prompt-visible
  `group_activity_window` fields are whitelisted to source, window bounds, and
  semantic labels only.
- Implemented reflection-cycle attachment in
  `src/kazusa_ai_chatbot/reflection_cycle/worker.py` and
  `src/kazusa_ai_chatbot/service.py`: group review runs after hourly
  reflection on the existing `REFLECTION_WORKER_INTERVAL_SECONDS` cadence and
  receives the live adapter registry provider so selected speech can dispatch
  through the bound source group.
- The default standalone self-cognition worker collector does not collect
  group review cases, so no second group-review cadence was introduced.
- Documentation updated:
  `src/kazusa_ai_chatbot/reflection_cycle/README.md` and
  `src/kazusa_ai_chatbot/self_cognition/README.md`.

Verification commands and results:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\reflection_cycle\activity_windows.py src\kazusa_ai_chatbot\reflection_cycle\worker.py src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\kazusa_ai_chatbot\service.py tests\test_reflection_cycle_activity_windows.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_tracking.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_service.py tests\test_self_cognition_delivery_target.py tests\test_self_cognition_integration.py
```

- Result: exit code 0.

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_activity_windows.py -q
```

- Result: 4 passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_self_cognition_delivery_target.py -q
```

- Result: 8 passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_self_cognition_group_review_source.py -q
```

- Result: 5 passed.
- Prompt/source-packet render check: passed by
  `test_group_review_source_packet_uses_active_group_review_contract`; it
  verifies active group-review wording, rendered group-review trigger,
  same-channel actionability, prompt-visible semantic labels, no private target
  leakage, and no empty group-window section for non-group sources.

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_stage1c_worker.py -q
```

- Result: 9 passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_self_cognition_integration.py -q
```

- Result: 26 passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_stage1c_service.py -q
```

- Result: 6 passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_prompt_contracts.py -q
```

- Result: 10 passed.

```powershell
venv\Scripts\python.exe -m pytest tests\test_self_cognition_tracking.py -q
```

- Result: 42 passed.

```powershell
git diff --check -- src\kazusa_ai_chatbot\reflection_cycle src\kazusa_ai_chatbot\self_cognition src\kazusa_ai_chatbot\service.py tests
```

- Result: exit code 0; only Git line-ending warnings were printed.

Final archive-state verification:

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_activity_windows.py tests\test_self_cognition_delivery_target.py tests\test_self_cognition_group_review_source.py tests\test_reflection_cycle_stage1c_worker.py tests\test_self_cognition_integration.py tests\test_reflection_cycle_stage1c_service.py tests\test_reflection_cycle_prompt_contracts.py tests\test_self_cognition_tracking.py -q
```

- Result: 110 passed.

```powershell
git diff --check -- development_plans src\kazusa_ai_chatbot\reflection_cycle src\kazusa_ai_chatbot\self_cognition src\kazusa_ai_chatbot\service.py tests
```

- Result: exit code 0; only Git line-ending warnings were printed.

### 2026-05-18 Independent Code Review

- Reviewer mode: main agent independent review after worker implementation.
- Inputs reviewed: approved plan, production diff, focused tests, subsystem
  READMEs, and verification outputs.
- Findings:
  1. Blocking, fixed: group review was briefly reachable from the default
     self-cognition worker collector, which would have created a second cadence.
     Fixed by keeping group-review collection only on the reflection worker
     path.
  2. Blocking, fixed: reflection-attached delivery initially lacked the live
     adapter-registry provider, so selected speech would have failed delivery
     instead of sending to the bound source group. Fixed by threading the
     provider from service startup through the reflection worker hook.
  3. Blocking, fixed: prompt-facing `group_activity_window` initially could
     pass through arbitrary caller keys and non-group packets rendered an empty
     group-window section. Fixed by whitelisting prompt-visible group-window
     fields and rendering the section only when present.
- Review result: approved after fixes and verification.
- Residual risks: the existing self-cognition route/action-spec mismatch from
  Stage 0 remains intentionally out of scope; group review still depends on
  existing adapter registration being available at delivery time. Fallback and
  retry remain intentionally out of scope. Adapter channel availability is
  checked before write-ahead delivery persistence.

### 2026-05-18 Stage 6 Cleanup Evidence

- Cleanup command verified the resolved absolute target path before recursive
  deletion, then removed
  `experiments/reflection_attached_group_self_cognition_probe/`.
- Removed the experiment-specific unignore rules from `experiments/.gitignore`.
- Cleanup verification:

```powershell
git status --short -- experiments
```

- Result after index refresh: no output.

```powershell
rg -n "reflection_attached_group_self_cognition_probe" experiments\.gitignore
```

- Result: exit code 1, no matches.

```powershell
Test-Path -LiteralPath 'experiments\reflection_attached_group_self_cognition_probe'
```

- Result: `False`.

### 2026-05-18 Surfaced Issue Fixes

- Fixed group direct-address labeling so a group message is only direct-addressed
  when the active character identity is targeted.
- Fixed group review case ordering so `max_cases` keeps the newest 15-minute
  activity windows.
- Added adapter channel availability preflight before dispatcher write-ahead
  persistence, with remote, Discord, and NapCat capability checks. Fallback and
  retry remain out of scope.
- Verification:

```powershell
venv\Scripts\python -m pytest tests\test_reflection_cycle_activity_windows.py::test_group_activity_window_direct_address_requires_character_identity tests\test_self_cognition_group_review_source.py::test_collect_group_chat_review_cases_prefers_newest_windows tests\test_self_cognition_integration.py::test_worker_channel_capability_failure_blocks_before_history_write -q
```

- Result: 3 passed.

```powershell
venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py::test_remote_http_adapter_posts_send_message_payload tests\test_runtime_adapter_registration.py::test_remote_http_adapter_posts_send_message_capability_payload -q
```

- Result: 2 passed.

```powershell
venv\Scripts\python -m pytest tests\test_reflection_cycle_activity_windows.py tests\test_self_cognition_group_review_source.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_service.py tests\test_self_cognition_delivery_target.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py tests\test_runtime_adapter_registration.py tests\test_dispatcher_event_logging.py tests\test_dispatcher_send_message_result.py tests\test_delivery_mentions.py -q
```

- Result: 157 passed.

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\activity_windows.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\delivery.py src\kazusa_ai_chatbot\dispatcher\adapter_iface.py src\kazusa_ai_chatbot\dispatcher\handlers.py src\kazusa_ai_chatbot\dispatcher\remote_adapter.py src\adapters\napcat_qq_adapter.py src\adapters\discord_adapter.py tests\test_reflection_cycle_activity_windows.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_integration.py tests\test_runtime_adapter_registration.py
```

- Result: exit code 0.

```powershell
git diff --check
```

- Result: exit code 0; only Git line-ending warnings were printed.
