# multi source cognition architecture stage 10 permissioned proactive output plan

## Summary

- Goal: Define a permissioned proactive-output path that turns approved
  cognition previews into auditable outbox records and transport sends only
  after explicit permission, target validation, adapter availability, and quiet
  hour checks pass.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `database-data-pull`, `py-style`,
  `test-style-and-execution`, and `cjk-safety` before editing Python files that
  contain CJK prompt text.
- Overall cutover strategy: migration. Start with preview/outbox dry-run
  records, then enable transport only behind an explicit deterministic
  permission record and a manual smoke gate. No autonomous contact is enabled
  by default.
- Highest-risk areas: sending without permission, wrong target, private thought
  leakage, scheduler bypass, duplicate sends, live `/chat` regression, and
  treating generated previews as user instructions.
- Acceptance criteria: proactive output has explicit permission, outbox,
  transport audit, duplicate-suppression, and rollback contracts; `/chat`
  regression gates pass; no send path runs without an approved permission
  fixture and test evidence.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: this draft is blocked until Stage 09 completes and records execution
evidence. Do not approve or execute Stage 10 while the parent ledger row for
`stage_09` is not `completed`. Because this stage can contact users, final
approval must be explicit and separate from draft creation.

## Context

Earlier stages make multiple trigger and input sources enter cognition without
changing current `/chat` behavior. Stage 10 is the first stage that may create
outward output not directly caused by the current user message. The top-level
architecture requires explicit permission, dispatcher or scheduler validation,
adapter availability, and auditability before any autonomous contact.

This plan must not turn reflection, internal thought, or media perception into
automatic messaging. A cognition result can only become a candidate preview.
Deterministic policy decides whether that preview can become an outbox item and
whether the outbox item can be sent.

## Stage Handoff

### From Stage 09

Stage 10 expects these completed artifacts:

- multimodal episode support with raw media excluded from prompts;
- reflection and internal-thought dry-run audit patterns from Stages 07 and 08;
- Stage 06 origin policy evidence that non-chat writes are denied by default;
- current `/chat` regression evidence after source expansion;
- parent ledger row for `stage_09` set to `completed`.

Before approval, replace this paragraph with exact Stage 09 branch, commit, and
verification results from Stage 09 `Execution Evidence`, then rerun the plan
self-review.

### To Later Work

After Stage 10, later plans can rely on:

- proactive output permission records and outbox records being separate from
  normal conversation history;
- transport sends being auditable and duplicate-suppressed;
- previews remaining unsent until deterministic policy approves them;
- `/chat` regression gates still protecting normal user-message behavior.

Later plans must not broaden permission semantics without a new approval.

## Mandatory Skills

- `development-plan-writing`: preserve high-risk staged lifecycle and explicit
  approval gates.
- `local-llm-architecture`: keep LLMs responsible for preview wording only;
  deterministic code owns permission, target, limits, and execution.
- `no-prepost-user-input`: do not infer permissions or commitments from user
  text with local keyword rules.
- `database-data-pull`: inspect existing scheduler/outbox-like records before
  approving database writes.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing any Python file with CJK prompt strings.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-09 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not send any proactive message unless an explicit permission record exists
  for the exact platform, channel, target scope, trigger source, and output
  mode.
- Do not infer permission from user wording with keyword matching or from LLM
  stance fields.
- Do not put previews into normal conversation history until after transport
  confirms a send and the outbox record is marked sent.
- Do not send private internal thought. Only approved preview text may enter an
  outbox item.
- Do not bypass dispatcher/scheduler validation, adapter availability checks,
  quiet hours, duplicate-suppression, target validation, or audit writes.
- Do not add new live `/chat` LLM calls.
- Do not change normal `/chat` delivery, reply targeting, assistant
  persistence, or consolidation behavior.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.

## Must Do

- Inspect existing scheduler, dispatcher, delivery tracking, and conversation
  persistence contracts before final approval.
- Define explicit proactive permission records with platform, channel,
  target-scope, allowed trigger sources, allowed output modes, quiet-hour
  policy, expiry, and audit metadata.
- Define proactive preview records separately from outbox/send records.
- Define outbox records with idempotency key, status, target, preview source,
  transport attempt metadata, and failure reason.
- Add deterministic policy that refuses missing permission, expired permission,
  wrong target, quiet hours, unsupported adapter, duplicate idempotency key,
  unsafe output mode, and private/audit-only content.
- Add dry-run outbox tests that create no adapter sends.
- Add transport tests with a fake adapter only after permission and outbox
  policy tests pass.
- Add persistence tests proving sent proactive messages are marked with
  explicit proactive origin metadata and are not treated as user instructions.
- Run the full Stage 00 regression harness before any transport cutover.
- Require manual review before changing lifecycle to `completed`.

## Deferred

- Broad autonomous contact.
- Permission inference from chat content.
- Multi-recipient fanout.
- Media sends.
- LLM-based safety repair loops.
- Proactive reflection-origin durable writes outside the outbox audit path.
- User-facing permission-management UI.
- Cross-platform transport abstraction redesign.

## Cutover Policy

Overall strategy: migration.

| Area | Policy | Instruction |
|---|---|---|
| Permission records | migration | Add explicit records and tests before any outbox send path. |
| Preview records | compatible | Store previews separately from normal conversation history. |
| Outbox dry run | migration | Create audit-only outbox records first. No adapter send. |
| Transport send | migration | Enable only after permission, adapter, target, quiet-hour, duplicate, and smoke gates pass. |
| `/chat` | compatible | No new calls, no delivery changes, no persistence changes for normal user-message turns. |

Rollback path: disable proactive transport execution, leave audit/outbox records
inspectable, and keep normal `/chat` path unchanged. If transport sends were
enabled, mark pending outbox rows cancelled through the approved operational
script named by the final approved plan. Do not delete audit history ad hoc.

## Agent Autonomy Boundaries

Allowed implementation choices before approval:

- none for code. This draft is not executable until Stage 09 evidence and
  database inspection are recorded.

Allowed implementation choices after approval:

- local test helper names;
- local variable names;
- assertion ordering.

Not allowed:

- inventing permission semantics, database collection names, adapter APIs,
  background worker scheduling, retry policy, or outbox status values outside
  this plan;
- adding send paths before dry-run outbox tests pass;
- adding fallback sends, direct adapter calls, or bypasses around dispatcher
  validation;
- adding keyword permission extraction;
- changing normal `/chat` service behavior;
- adding raising-only helpers or pass-through wrappers.

If implementation reveals an unlisted transport or database dependency, stop
and update this plan before continuing.

## Target State

The proactive path is:

```text
approved cognition preview
-> proactive permission policy
-> proactive preview record
-> outbox dry-run record
-> transport send only after all deterministic gates pass
-> sent audit record
-> optional assistant conversation row marked proactive_sent
```

The normal `/chat` path is unchanged. A proactive sent row must not be consumed
as a user instruction or stored as if a user initiated the turn.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Permission | Explicit deterministic record only. | Prevents LLM or keyword inference from authorizing contact. |
| Preview vs send | Separate preview, outbox, and transport records. | Keeps unsent cognition output out of public history. |
| Transport | Fake adapter first, real adapter only after smoke gate. | High-risk user contact needs staged evidence. |
| Idempotency | Required key per source preview and target. | Prevents duplicate proactive messages. |
| `/chat` | No behavior change. | User-message response remains the baseline. |

## Change Surface

Final approval must confirm exact collection names and adapter APIs after
Stage 09. This draft proposes the following target surface:

### Create

- `src/kazusa_ai_chatbot/proactive_output/contracts.py` — permission,
  preview, outbox, and audit TypedDicts.
- `src/kazusa_ai_chatbot/proactive_output/policy.py` — deterministic
  permission and send gating.
- `src/kazusa_ai_chatbot/proactive_output/outbox.py` — outbox persistence
  boundary or in-memory dry-run boundary, as approved after DB inspection.
- `tests/test_multi_source_cognition_stage_10_proactive_policy.py`
- `tests/test_multi_source_cognition_stage_10_proactive_outbox.py`

### Modify

- exact dispatcher/adapter or scheduler integration files to be named after
  Stage 09 and database inspection;
- lifecycle rows in the parent plan and registry after completion only.

### Keep

- `src/kazusa_ai_chatbot/service.py` normal `/chat` response path unless final
  approval names a surgical persistence marker for sent proactive rows;
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2*.py`
- cognition prompt files;
- RAG files;
- consolidation policy files except explicit origin metadata for
  `proactive_sent` if a later approved step names it.

## Implementation Order

This draft is not approval to implement. Before approval:

1. Reread Stage 09 `Execution Evidence`.
2. Inspect existing scheduler, dispatcher, delivery tracking, and conversation
   persistence contracts.
3. Update Change Surface with exact files and exact database collection names.
4. Add final policy, outbox, transport, rollback, and verification commands.
5. Rerun plan self-review and obtain explicit user approval.

After approval, the intended execution order is:

1. Add permission policy unit tests.
2. Implement permission policy contracts.
3. Add outbox dry-run tests.
4. Implement outbox dry-run persistence boundary.
5. Add fake-adapter transport tests.
6. Implement transport send boundary.
7. Add sent-audit and conversation-row tests.
8. Run full verification including Stage 00.
9. Record evidence and request review before merge.

## Progress Checklist

- [ ] Stage 1 - prerequisite and database evidence carried forward.
  - Covers: pre-approval Steps 1-2.
  - Verify: Stage 09 row is `completed`; DB/dispatcher inspection notes are
    recorded.
  - Evidence: Stage 09 branch, commit, tests, and inspection summary recorded.
  - Handoff: next agent updates final Change Surface.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - final executable contract approved.
  - Covers: pre-approval Steps 3-5.
  - Verify: exact files, collections, adapter APIs, rollback script, and test
    commands are named.
  - Evidence: user approval recorded.
  - Handoff: implementation may start at Stage 3.
  - Sign-off: `<agent/date>` after approval.
- [ ] Stage 3 - permission policy implemented.
  - Covers: approved Steps 1-2.
  - Verify: policy unit tests pass after expected red failure.
  - Evidence: red/green output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 - outbox dry-run implemented.
  - Covers: approved Steps 3-4.
  - Verify: outbox dry-run tests pass and no adapter send occurs.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 - transport and sent-audit implemented.
  - Covers: approved Steps 5-7.
  - Verify: fake-adapter and sent-audit tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 6.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 6 - full verification and manual review complete.
  - Covers: approved Steps 8-9.
  - Verify: every Verification command passes and manual review is recorded.
  - Evidence: command output and review result recorded.
  - Handoff: no later stage in this parent plan.
  - Sign-off: `<agent/date>` after verification.

## Verification

This draft names minimum gates. Final approval must replace the placeholder
transport/database commands with exact commands after inspection.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\proactive_output\contracts.py src\kazusa_ai_chatbot\proactive_output\policy.py src\kazusa_ai_chatbot\proactive_output\outbox.py tests\test_multi_source_cognition_stage_10_proactive_policy.py tests\test_multi_source_cognition_stage_10_proactive_outbox.py`

### Static Greps

- `rg -n "send_message|adapter\\.send|save_conversation|dispatcher\\.dispatch" src\kazusa_ai_chatbot\proactive_output`

  Expected result before transport stage: matches only in tests or explicitly
  approved fake-adapter boundary. After transport approval, matches must be
  limited to the named transport module.

- `rg -n "proactive|scheduled_action_request|proactive_sent" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`

  Expected result: no matches unless final approval names a surgical sent-row
  persistence marker. Normal `/chat` response path must not branch on proactive
  state.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_10_proactive_policy.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_10_proactive_outbox.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`
- `venv\Scripts\python -m pytest`

### Manual Review

- Inspect the outbox dry-run fixture and fake-adapter transport audit.
- Confirm no real adapter send occurs before the approved transport gate.
- Confirm permission records are explicit and not inferred from user text.

## Acceptance Criteria

Stage 10 is complete when:

- permission, preview, outbox, and audit contracts exist;
- missing, expired, wrong-target, quiet-hour, duplicate, unsupported-adapter,
  unsafe-mode, and private-content cases are denied deterministically;
- dry-run outbox creates no adapter send;
- approved fake-adapter transport writes auditable sent metadata;
- proactive sent rows, if enabled, are marked with explicit origin and are not
  treated as user instructions;
- Stage 00 full regression and full deterministic suite pass;
- manual review approves the send boundary before merge.

## Plan Self-Review

Draft self-review on 2026-05-10:

- **Coverage:** parent Stage 10 scope maps to permission, preview, outbox,
  transport, audit, rollback, and regression gates.
- **Placeholder scan:** exact Stage 09 evidence, DB collection names, adapter
  APIs, and final transport commands remain blocked until prerequisite
  inspection; this draft is intentionally not executable.
- **Contract consistency:** permission-first proactive output matches the
  top-level architecture and does not authorize autonomous contact by default.
- **Granularity:** checkpoints separate pre-approval evidence, final contract,
  policy, dry-run outbox, transport, and review.
- **Verification:** permission denial, no-send dry run, fake transport,
  full-suite regression, and manual review are explicit.

## Execution Handoff

Intended execution mode after final approval: sequential implementation on a
feature branch forked from post-Stage-09 `main`.

Blocked next action: wait for Stage 09 completion evidence, inspect
scheduler/dispatcher/delivery/database contracts, then update this draft into a
final executable plan for explicit user approval.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Message sends without permission | Explicit permission record required | Policy denial tests |
| Wrong target receives message | Target validation before outbox/send | Policy tests and fake adapter audit |
| Duplicate proactive send | Idempotency key and outbox status | Outbox tests |
| Private thought leaks | Only approved preview text can send | Policy tests |
| `/chat` regresses | No normal response-path changes | Stage 00 and full suite |
| Audit loss after rollback | Keep records inspectable; cancel pending rows | Manual review and rollback notes |

## Completion Artifact Contract

When Stage 10 is complete, these artifacts must exist or be updated:

- proactive permission, preview, outbox, and audit contracts;
- deterministic proactive policy tests;
- outbox dry-run and fake-adapter transport tests;
- explicit rollback/cancel procedure for pending outbox rows;
- parent ledger row for `stage_10` flipped to `completed`;
- registry row flipped to `completed | completed`;
- execution evidence in this plan naming branch, commit, checks, manual review,
  and sign-off.

The artifact must not include inferred permissions, broad autonomous contact,
normal `/chat` response-path changes, direct adapter bypasses, or hidden
transport sends.

## Execution Evidence

Record after implementation:

- Stage 09 evidence reread:
- Database and dispatcher inspection:
- User approval for final executable plan:
- Branch:
- Commit:
- Static compile:
- Static greps:
- Focused tests:
- Prior stage regression gates:
- Full suite:
- Manual review:
- Completion diff review:
- Lifecycle records:
- Sign-off:
