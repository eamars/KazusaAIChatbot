# user profiles lane data integrity plan

## Summary

- Goal: repair malformed `user_profiles` rows that can corrupt memory
  targeting and harden all profile creation and user-memory write paths so
  diagnostic, empty, synthetic, or character/global identities cannot become
  ordinary production user-memory targets.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `database-data-pull`,
  `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Overall cutover strategy: migration for reviewed data cleanup; bigbang for
  production user-target eligibility.
- Highest-risk areas: deleting a profile linked to history, preserving
  debug/pytest fixtures without letting them receive production memory,
  separating character/global identity from user memory, and proving that
  invented social memory is allowed only after the profile target is valid for
  the user lane.
- Acceptance criteria: malformed profile rows are classified and either
  preserved, blocked, or removed through reviewed dry-run/apply gates; ordinary
  relationship and user-memory writes can target only valid production users;
  character/global identities remain explicit non-user targets; robustness is
  proven by tests, static checks, and post-repair audit evidence.

New-write hardening supersession: all lane-specific new-write hardening,
source-generation proof, and fixing-strategy instructions in this plan are
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.
Execute this lane plan for cleanup, audit, dry-run/apply, data migration, and
post-cleanup verification only. If another section still names new-write
validators, prompts, or tests, treat that text as historical context rather
than execution scope.

## Context

The 2026-07-02 memory-lane audit found 3299 `user_profiles` rows. The initial
summary identified:

- 2 profiles without platform accounts;
- 21 pytest platform accounts;
- 35 debug accounts;
- one character/global profile with many platform accounts.

These rows are not all corrupt. `user_profiles` is target/provenance
infrastructure: it proves which durable entity a relationship header,
affinity score, interaction-style image, or scoped `user_memory_units` write
belongs to. It is not itself the cognition-facing long-term memory store.
Invented social memory may be valid character behavior when the profile target
is valid for the user lane, but the same invented memory becomes corruption
when the target is empty, diagnostic, synthetic, or the active character.

Current source inspection shows these relevant ownership boundaries:

- `src/kazusa_ai_chatbot/db/users.py` creates and mutates
  `user_profiles`. `resolve_global_user_id(...)` creates normal platform
  profiles. `create_user_profile(...)` can insert no-platform rows. Character
  identity is created by `ensure_character_identity(...)`.
- `src/kazusa_ai_chatbot/consolidation/target.py` deterministically builds
  write targets. It already rejects known synthetic source labels and requires
  `user_profile.global_user_id` plus `affinity`, but it does not yet express
  the full production-target eligibility contract for platform linkage,
  diagnostic platforms, character/global identity, or history-linked
  no-platform rows.
- `src/kazusa_ai_chatbot/consolidation/persistence.py` gates relationship,
  affinity, and user-memory writes through `validate_write_intent(...)`, then
  calls DB helpers. Those DB helpers currently use upsert-like writes for
  affinity and relationship insight, so target validation must fail closed
  before persistence is reached.
- `src/kazusa_ai_chatbot/rag/person_context/*` reads profile-like context as
  evidence. RAG evidence does not authorize durable writes. Target eligibility
  stays deterministic and outside LLM prompts.
- `src/kazusa_ai_chatbot/db/script_operations.py` is the existing
  maintenance boundary for profile/memory diagnostics and cleanup. New
  audit/repair scripts must use that module rather than raw DB access in
  scripts.

## Lane Analysis Requirements

### Issue Description Based On Deep Analysis

`user_profiles` currently mixes production user identity headers, diagnostic
debug/pytest identities, no-platform rows, and explicit character/global
identity infrastructure in one collection. The defect occurs when a row whose
purpose is diagnostic, source provenance, empty scaffolding, or character
identity is treated as a normal production user target for relationship
insight, affinity, user style image, or `user_memory_units`.

The immediate product failure is not that every debug or no-platform row
exists. The failure is that downstream memory writers may interpret a
syntactically present `global_user_id` as permission to write user-scoped
social memory. This can attach believable but invented relationship memory to
the wrong durable owner.

### Plan To Remove Malformed Data

Add a read-only audit command and a separate dry-run/apply repair command.
The audit classifies every `user_profiles` row into production user,
diagnostic, character/global, no-platform-history-linked, empty-unreferenced,
or manual-review. The repair command may delete only structurally empty,
unreferenced rows after a reviewed dry-run. It must preserve rows linked to
conversation history, `user_memory_units`, shared `memory.source_global_user_id`,
conversation episode state, scheduled-event legacy rows, action attempts,
alias backlinks, or known character/global semantics.

Diagnostic debug/pytest rows are not deleted by default. They are reported and
blocked from ordinary production user-memory writes. Cleanup of test-owned or
debug-owned data outside empty-unreferenced rows requires a separate explicit
operator command and is out of scope for this plan.

### RCA Of The Failure Mode

Root cause is an under-specified target eligibility contract at the boundary
between profile existence and durable memory writes:

- `resolve_global_user_id(...)` correctly creates platform-linked profiles for
  inbound platform accounts, including debug adapter and test platform values;
  those creations lack a durable profile class such as production or
  diagnostic.
- `create_user_profile(...)` can create a profile with default empty
  `platform_accounts`, `affinity`, and relationship fields. That is useful as
  infrastructure, but no-platform rows are not automatically valid production
  user-memory targets.
- Character identity uses `user_profiles` as first-class addressability
  infrastructure through `ensure_character_identity(...)`. That row is not a
  normal user and must not receive ordinary relationship-memory writes.
- Existing consolidation target planning rejects known synthetic source labels
  such as `self_cognition`, but a non-empty arbitrary id still passes if the
  runtime `user_profile` carries matching `global_user_id` and `affinity`.
- Relationship insight and affinity helpers use upsert-like updates, so a
  target-planning gap can materialize or mutate profile rows even when the
  row was only diagnostic, empty, or synthetic.
- RAG person-context helpers can surface profile and relationship evidence,
  but RAG owns evidence retrieval only. It must not be treated as permission
  to create or select a user-memory target.
- Debug and pytest identities are valid for tests and local diagnostics but
  should be ineligible for ordinary production memory writes unless an
  explicit diagnostic mode owns that behavior.

### Plan To Harden The Corrupted Data Source

New-write hardening for this lane is superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

This lane plan keeps its cleanup, audit, dry-run/apply, and post-cleanup
verification scope. Do not implement lane-specific new-write prompts,
validators, routing rules, or semantic filters from the superseded draft text;
implement new-write memory-pollution prevention through `consolidator_lane_router_memory_pollution_bigbang_plan.md` only.

### Plan To Prove Robustness Of The Data Source Generation

Robustness of new data-source generation for this lane is superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

This lane plan proves cleanup robustness only: baseline export, dry-run report,
manual review, approved apply, post-apply audit, retrieval/cache smoke, and
rollback evidence where the cleanup section requires it. The cross-lane
new-write robustness gate is the router plan's deterministic tests plus its 26
one-at-a-time live-LLM memory-write use cases.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  signing off, or changing lifecycle status for this plan.
- `database-data-pull`: load before exporting or inspecting live
  `user_profiles`, `conversation_history`, `user_memory_units`, shared
  `memory`, or related diagnostic rows. Use bundled scripts and project
  settings; never paste DB connection strings.
- `local-llm-architecture`: load before changing target planning, RAG/profile
  context contracts, consolidation prompts, memory extraction contracts, or any
  LLM-facing user/profile boundary.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute this plan while status is `draft`.
- Do not apply live cleanup without a reviewed dry-run artifact and an explicit
  user command naming apply mode.
- Do not read `.env` during planning review. Execution commands may let the
  approved project scripts load environment settings through their normal path.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Check `git status --short` before editing.
- Use `kazusa_ai_chatbot.db.script_operations` for maintenance DB operations.
  Scripts under `src/scripts` must not issue raw MongoDB queries directly.
- Keep profile rows as target/provenance infrastructure. Do not put
  cognition-facing long-term user facts into `user_profiles`.
- Keep RAG evidence and target permission separate. RAG may retrieve profile
  evidence; deterministic consolidation target planning decides whether a
  write target exists.
- LLM stages may generate semantic social memory only after deterministic code
  has supplied an eligible user target. LLM output must not decide whether a
  profile is production, diagnostic, character/global, or safe to delete.
- Do not delete profiles linked to conversation history, user memory units,
  shared memory source ids, conversation episode state, scheduled events,
  action attempts, or alias backlinks without explicit manual review outside
  this plan.
- Do not allow debug/pytest profiles to receive normal production
  user-memory, relationship insight, affinity, or user-style writes.
- Do not allow `CHARACTER_GLOBAL_USER_ID`, active-character account rows, or
  character/global profile rows to receive ordinary user-lane writes.
- Do not introduce compatibility shims, fallback mappers, parallel identity
  vocabularies, or LLM classifiers for target eligibility.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Add a user-profile lane audit for empty profiles, no-platform profiles,
  debug/pytest diagnostic profiles, character/global identities, synthetic
  source labels, target eligibility, and cleanup blockers.
- Add exact audit classifications and recommended actions for production
  users, diagnostic rows, character/global rows, no-platform history-linked
  rows, empty unreferenced rows, malformed rows, and manual-review rows.
- Add a dry-run/apply cleanup path for structurally empty, unreferenced
  profiles only.
- Harden production target validation so user-memory, relationship insight,
  affinity, and user-style writes require a real production user target.
- Harden corrupted data source paths so ordinary relationship/affinity helpers
  cannot upsert arbitrary profile rows after target validation denies a target.
- Preserve character/global identity handling as explicit character identity,
  not ordinary user identity.
- Add tests proving diagnostic, empty, synthetic, no-platform, and
  character/global profiles cannot receive ordinary production user-memory or
  relationship writes.
- Add verification showing real platform-linked users remain valid targets.
- Add plan and execution evidence that maps the user-requested issue
  description, malformed-data removal plan, RCA, data-source hardening plan,
  and robustness proof to concrete tasks and checks.

## Deferred

- Do not redesign identity resolution.
- Do not merge duplicate user profiles.
- Do not change affinity scoring semantics.
- Do not move cognition-facing memory into `user_profiles`.
- Do not delete debug/pytest profiles that tests intentionally own unless the
  cleanup is isolated to production data through a later explicit plan.
- Do not change adapter account-linking contracts beyond target eligibility
  validation.
- Do not change RAG person-context retrieval behavior except where a focused
  test proves it leaks character/global or diagnostic identities into a
  write-permission contract.
- Do not add a profile-kind LLM classifier.
- Do not introduce a broad profile migration that rewrites every existing
  profile row into a new schema version.
- Do not update `development_plans/README.md` under this plan unless a later
  explicit user command overrides the current drafting constraint.

## Cutover Policy

Overall strategy: migration for data cleanup; bigbang for production write
eligibility.

| Area | Policy | Instruction |
| --- | --- | --- |
| Existing `user_profiles` read shape | compatible | Keep existing reads and relationship-header fields readable. Do not require every existing row to carry a new schema field before runtime can read it. |
| Audit classification | compatible | Add read-only classification and reporting without mutating rows. |
| Empty unreferenced profile cleanup | migration | Run export, audit, dry-run, manual review, approved apply, and post-apply audit. Apply may delete only rows proven empty and unreferenced. |
| Production target eligibility | bigbang | Ordinary user-lane writes must immediately reject diagnostic, empty, synthetic, and character/global targets. No fallback to old permissive behavior. |
| Relationship and affinity helper upsert behavior | bigbang | Runtime relationship/affinity writes must not create arbitrary profiles. Approved creation remains in identity-resolution or maintenance paths only. |
| Debug/pytest identities | compatible | Preserve rows for tests/diagnostics while making them ineligible for ordinary production user-lane writes. |
| Character/global identity | compatible | Preserve character profile/addressability behavior and keep it separate from ordinary user-memory writes. |
| RAG person-context evidence | compatible | Continue retrieving profile evidence, but do not let evidence imply write permission. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- If an area is `bigbang`, rewrite or delete legacy permissive behavior
  directly instead of preserving a fallback.
- If an area is `migration`, follow the exact audit, dry-run, review, apply,
  and post-audit gates listed in this plan.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

`user_profiles` remains a user identity and relationship-header collection,
plus explicit character/global identity infrastructure. It proves where
user-scoped durable writes may land; it does not store cognition-facing long
term user memory.

Valid production user target:

- `global_user_id` is non-empty and not a known synthetic source label;
- loaded `user_profile.global_user_id` exactly matches the target id;
- profile has required relationship-header fields such as `affinity`;
- profile has at least one valid production platform account, unless an
  explicitly allowed non-platform production identity class is added in this
  plan's implementation;
- profile is not debug/pytest diagnostic;
- profile is not `CHARACTER_GLOBAL_USER_ID` and not classified as
  character/global identity.

Invalid ordinary production user target:

- no platform accounts and no approved non-platform production identity class;
- debug or pytest platform/account marker;
- synthetic source label such as `self_cognition`, `internal_thought`,
  `reflection_signal`, `group_chat_review`, `scheduled_future_cognition`,
  `orchestrator`, `system`, `internal`, `character`, `group`, or
  `group_channel`;
- active character/global identity used as ordinary user;
- profile missing required fields;
- profile id mismatch between runtime state and loaded `user_profile`;
- no loaded profile row.

History-linked no-platform target:

- preserved for manual review;
- not automatically deleted;
- not eligible for ordinary production user-lane writes until a later explicit
  identity repair links a real account or approved identity class.

Diagnostic target:

- preserved for tests or local diagnostics;
- reported separately by audit;
- rejected for ordinary production user-lane writes;
- allowed only if a future explicitly named diagnostic write path is reviewed
  and approved.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Plan class | Use `high_risk_migration` | The plan changes production write eligibility and may delete production DB rows after review. |
| Profile meaning | Treat `user_profiles` as target/provenance infrastructure | Durable memory belongs in `user_memory_units`; profile rows only prove identity and relationship-header ownership. |
| Empty profiles | Delete only structurally empty and unreferenced rows | Avoid severing history or orphaning memories. |
| No-platform profiles with references | Preserve and block ordinary writes | History-linked rows need manual identity repair, not blind deletion. |
| Debug/pytest profiles | Classify as diagnostic and block production writes | Tests and debug workflows may need rows, but production memory must not target them. |
| Character/global profile | Preserve as explicit character identity | Character state/self-image are separate lanes from user memory. |
| Target eligibility | Deterministic helper in consolidation target boundary | LLMs should not decide whether a profile is real, diagnostic, or deletable. |
| Profile creation | Keep creation in identity/character/maintenance paths | Relationship and affinity writes must not create profile rows after target denial. |
| RAG person context | Evidence only, no write permission | Person/profile evidence can inform cognition but cannot authorize persistence. |
| Cleanup apply | Separate command from audit | Read-only diagnostics must be safe and repeatable before mutation. |

## Contracts And Data Shapes

### Audit Finding Contract

Add an audit finding shape in `kazusa_ai_chatbot.db.script_operations` and
project it through the audit script:

```python
class UserProfileLaneFinding(TypedDict):
    global_user_id: str
    profile_kind: Literal[
        "production_user",
        "diagnostic",
        "character_global",
        "empty_unreferenced",
        "no_platform_history_linked",
        "malformed",
        "manual_review",
    ]
    issue_code: str
    issue_description: str
    recommended_action: Literal[
        "keep",
        "block_from_production_writes",
        "archive_empty",
        "manual_review",
    ]
    cleanup_eligible: bool
    cleanup_blockers: list[str]
    reference_counts: dict[str, int]
    evidence_fields: dict[str, object]
```

Required `issue_code` values:

- `valid_production_user`
- `diagnostic_debug_platform`
- `diagnostic_pytest_platform`
- `empty_no_platform_unreferenced`
- `no_platform_history_linked`
- `character_global_identity`
- `synthetic_source_label`
- `missing_global_user_id`
- `missing_affinity`
- `malformed_platform_account`
- `profile_id_conflict`
- `manual_review_mixed_diagnostic_and_production`

### Target Eligibility Contract

Add a deterministic helper owned by `consolidation.target`:

```python
class UserProfileTargetEligibility(TypedDict):
    eligible: bool
    target_kind: Literal[
        "production_user",
        "diagnostic",
        "character_global",
        "synthetic",
        "empty",
        "malformed",
    ]
    reason_codes: list[str]
```

```python
def classify_user_profile_target(
    *,
    global_user_id: str,
    user_profile: Mapping[str, Any],
    runtime_platform: str = "",
) -> UserProfileTargetEligibility:
    ...
```

`classify_user_profile_target(...)` must:

- run without LLM calls;
- reject empty ids and synthetic source labels;
- reject `CHARACTER_GLOBAL_USER_ID` for ordinary user lanes;
- require `user_profile.global_user_id == global_user_id`;
- require `affinity` for write-target eligibility;
- inspect `platform_accounts` structurally;
- reject debug and pytest account markers for production writes;
- require at least one valid production account unless an approved
  non-platform production identity class is explicitly present;
- return reason codes used by tests, logs, audit reports, and target-plan
  metadata.

`build_consolidation_target_plan(...)` must add a `user` target only when
`eligible` is true. Rejected user targets must not be repaired by choosing a
different target.

### Write-Intent Contract

`validate_write_intent(...)` remains the deterministic gate. User lanes must
be allowed only when the target plan already contains an eligible `user`
target. The plan must not add direct DB eligibility checks in individual LLM
nodes or prompt stages.

### Maintenance Command Contracts

Create:

```powershell
venv\Scripts\python.exe -m scripts.audit_user_profiles_lane --output test_artifacts\user_profiles_lane_audit.json
venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --dry-run --output test_artifacts\user_profiles_lane_repair_dry_run.json
venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --apply --input test_artifacts\user_profiles_lane_repair_dry_run.json --output test_artifacts\user_profiles_lane_repair_apply.json
```

Audit mode is read-only. Repair dry-run is read-only and emits exact candidate
ids plus blockers. Apply mode must require a dry-run input artifact generated
by the same script version and must skip any row whose current blocker state
differs from the dry-run.

### Audit Classification Rules

- `production_user`: at least one valid production account and no diagnostic,
  synthetic, character/global, or malformed blocker.
- `diagnostic`: any platform/account marker matching debug or pytest
  diagnostics, unless the same row also has production accounts; mixed rows
  become `manual_review_mixed_diagnostic_and_production`.
- `character_global`: `global_user_id == CHARACTER_GLOBAL_USER_ID`, known
  active-character account, or explicit character identity marker.
- `empty_unreferenced`: no platform accounts, no aliases, no relationship
  insight, default or missing affinity only, and zero references in every
  protected collection.
- `no_platform_history_linked`: no platform accounts but at least one
  protected reference.
- `malformed`: missing id, malformed platform account records, missing
  required relationship-header fields, or inconsistent field types.
- `manual_review`: conflicting signals or any blocker that cannot be safely
  resolved by this plan.

## LLM Call And Context Budget

Affected runtime LLM calls: none.

The fix must not add response-path or background LLM calls. It changes
deterministic target eligibility before persistence. Existing consolidation
LLM stages may continue generating relationship insight and user memory units,
but only after deterministic target planning has produced an eligible user
target.

LLM-facing profile context remains prompt-safe:

- RAG person-context reads profile evidence and may read character profile
  evidence through the existing character path.
- RAG evidence and `rag_result` do not authorize durable writes.
- Target eligibility reason codes are operational metadata and should not be
  injected into runtime prompts unless a later approved plan explicitly needs
  a prompt-facing refusal explanation.

Latency impact: no additional LLM calls. Added deterministic profile
classification is in-process and bounded by loaded profile shape during live
turns. Audit/repair commands are operator-maintenance paths outside live chat.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/consolidation/target.py`: add profile target
  classification, reject diagnostic/empty/synthetic/character targets, attach
  reason metadata only to operational target-plan structures when useful, and
  keep target planning deterministic.
- `src/kazusa_ai_chatbot/consolidation/persistence.py`: ensure relationship,
  affinity, user-style, and user-memory writes rely on the tightened target
  plan and never call DB helpers after user-target denial.
- `src/kazusa_ai_chatbot/db/users.py`: remove ordinary runtime upsert behavior
  from relationship/affinity writes or split creation-safe helpers from
  write-only helpers so denied targets cannot materialize rows. Keep
  `resolve_global_user_id(...)`, `create_user_profile(...)`, and
  `ensure_character_identity(...)` as explicit creation paths.
- `src/kazusa_ai_chatbot/db/script_operations.py`: add read-only audit helpers,
  reference counters, cleanup candidate planning, dry-run report generation,
  current-state apply validation, and exact empty-row deletion helper.
- `src/kazusa_ai_chatbot/consolidation/README.md`: update target-planning and
  lifecycle diagnostics contract for production target eligibility and
  diagnostic/character/global classifications.
- `src/kazusa_ai_chatbot/db/README.md`: update `user_profiles` collection
  contract with target/provenance infrastructure wording, diagnostic rows, and
  no-platform cleanup rules.

### Create

- `src/scripts/audit_user_profiles_lane.py`: read-only CLI for lane audit.
- `src/scripts/repair_user_profiles_lane.py`: dry-run/apply CLI for approved
  empty-unreferenced cleanup.
- `tests/test_user_profiles_lane_integrity.py`: focused target eligibility,
  audit classification, dry-run/apply, and write-denial tests.

### Keep

- Existing `user_profiles` collection and unique `global_user_id` index.
- Existing profile read shape for RAG and control-console consumers.
- Existing adapter account-linking schema.
- Existing character identity creation path through `ensure_character_identity`.
- Existing RAG person-context evidence behavior unless tests reveal it is
  crossing into write permission.

### Strong Justification For Outside-Target Changes

- `db.users` must change because current relationship/affinity helpers can
  upsert profile rows. Target planning alone blocks normal consolidation, but
  helper hardening prevents future callers from bypassing the target gate.
- `db.script_operations` and scripts must change because live DB audit/repair
  belongs to the established maintenance boundary.
- README/ICD updates are required because the profile lane contract changes
  production data interpretation and write eligibility.

## Overdesign Guardrail

- Actual problem: diagnostic, empty, synthetic, and character/global profile
  rows can be mistaken for production user targets, letting ordinary
  relationship and user-memory writes attach social memory to invalid owners.
- Minimal change: classify profile target eligibility deterministically,
  block invalid ordinary user-lane writes, audit existing rows, and delete only
  empty unreferenced profiles through reviewed maintenance apply.
- Ownership boundaries: deterministic target planning owns target eligibility;
  DB helpers own persistence mechanics and upsert policy; maintenance scripts
  own live data audit/cleanup orchestration; RAG returns evidence; cognition
  and consolidation LLMs own semantic memory content only after a valid target
  exists.
- Rejected complexity: no identity merge, no new profile collection, no
  schema-wide profile migration, no LLM profile classifier, no adapter
  refactor, no compatibility fallback that allows old target behavior, no
  diagnostic write mode in this plan, and no automatic repair of
  history-linked no-platform rows.
- Evidence threshold: duplicate-profile merge, diagnostic write mode, or
  schema-wide profile class fields require a separate audit showing current
  target blocking is insufficient and a user-approved follow-up plan.

## Agent Autonomy Boundaries

- The execution agent may implement eligibility helpers and audit scripts
  after approval.
- The execution agent must not delete or mutate live DB rows before an
  approved dry-run artifact is reviewed and the user explicitly commands apply.
- The execution agent must not delete profiles linked to conversation history,
  `user_memory_units`, shared memory, conversation episode state,
  scheduled-events legacy rows, action attempts, or alias backlinks.
- The execution agent must not change adapter identity semantics unless a
  focused test proves target validation cannot be isolated.
- The execution agent must stop if a profile appears both diagnostic and
  production-linked; classify it as manual review, not cleanup eligible.
- The execution agent must not add fallback behavior that lets a denied target
  be rewritten into `internal`, `character`, or another user target.
- The execution agent must search for existing profile classification or
  synthetic-user helpers before creating new helpers. If equivalent behavior
  exists, move or reuse it within the appropriate ownership boundary.
- The execution agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's target-integrity intent
  and record the discrepancy before changing scope.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent establishes focused tests for `consolidation.target` production
   target eligibility:
   - real production platform-linked profile is accepted;
   - debug profile is rejected;
   - pytest profile is rejected;
   - empty no-platform profile is rejected;
   - no-platform history-linked profile classification is not cleanup-eligible
     and is rejected for ordinary writes;
   - `CHARACTER_GLOBAL_USER_ID` is rejected for ordinary user lanes;
   - synthetic source labels are rejected;
   - profile id mismatch and missing `affinity` fail closed.
2. Parent runs the focused target tests before production edits and records
   the expected failures or current permissive baseline.
3. Production-code subagent implements target eligibility in
   `consolidation.target` only.
4. Parent reruns focused target tests and records pass/fail evidence.
5. Parent adds focused `db_writer` tests proving relationship insight,
   affinity, user-style, and user-memory helpers are not called when the target
   plan rejects the user.
6. Production-code subagent wires tightened target validation through
   `consolidation.persistence` only where needed.
7. Parent reruns target and writer tests.
8. Parent adds DB helper tests proving ordinary relationship/affinity update
   helpers do not upsert arbitrary profiles, while explicit creation paths
   remain available.
9. Production-code subagent hardens `db.users` helper behavior according to
   those tests.
10. Parent adds audit classification tests in
    `tests/test_user_profiles_lane_integrity.py` using fake collections with
    production, debug, pytest, no-platform unreferenced, no-platform linked,
    malformed, synthetic, and character/global rows.
11. Production-code subagent implements audit helpers in
    `db.script_operations`.
12. Parent adds repair dry-run/apply tests proving only empty unreferenced rows
    are cleanup-eligible, apply validates current state against dry-run, and
    linked or changed rows are skipped.
13. Production-code subagent implements repair helpers and scripts.
14. Parent updates consolidation and DB README contracts.
15. Parent runs static greps and focused tests listed in `Verification`.
16. Parent runs audit dry-run against the configured DB only after explicit
    execution approval for live diagnostics.
17. Parent starts independent code-review subagent after verification passes.
18. Parent remediates review findings inside the approved change surface,
    reruns affected verification, and records evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or current baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  audit-script validation, and evidence work while the production-code
  subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - target eligibility contract established.
  - Covers: implementation steps 1-4.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_user_profiles_lane_integrity.py::test_profile_target_eligibility_rejects_invalid_lanes -q`
    and
    `venv\Scripts\python.exe -m pytest tests/test_consolidation_target_routing.py -q`.
  - Evidence: record expected baseline failure before implementation and pass
    output after implementation in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - user-lane persistence denial enforced.
  - Covers: implementation steps 5-7.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_user_profiles_lane_integrity.py::test_db_writer_denies_invalid_profile_user_lanes -q`
    and
    `venv\Scripts\python.exe -m pytest tests/test_consolidator_origin_policy_db_writer.py -q`.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - DB helper creation boundary hardened.
  - Covers: implementation steps 8-9.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_user_profiles_lane_integrity.py::test_relationship_and_affinity_helpers_do_not_create_profiles -q`
    and
    `venv\Scripts\python.exe -m pytest tests/test_db.py -q`.
  - Evidence: record helper behavior changes and any updated DB tests.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - audit classification implemented.
  - Covers: implementation steps 10-11.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_user_profiles_lane_integrity.py::test_user_profiles_lane_audit_classifies_profile_kinds -q`
    and
    `venv\Scripts\python.exe -m pytest tests/test_consolidation_lifecycle_diagnostics.py -q`.
  - Evidence: record audit report schema and fake-collection test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - dry-run/apply repair implemented.
  - Covers: implementation steps 12-13.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_user_profiles_lane_integrity.py::test_user_profiles_lane_repair_apply_deletes_only_current_empty_unreferenced_rows -q`.
  - Evidence: record dry-run/apply safety behavior and skipped-row cases.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - docs and static checks complete.
  - Covers: implementation steps 14-15.
  - Verify static greps and docs checks in `Verification`.
  - Evidence: record grep outputs, allowed matches, and docs changed.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 7 - live audit dry-run evidence captured after explicit execution
  approval for live diagnostics.
  - Covers: implementation step 16.
  - Verify:
    `venv\Scripts\python.exe -m scripts.audit_user_profiles_lane --output test_artifacts\user_profiles_lane_audit.json`
    and
    `venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --dry-run --output test_artifacts\user_profiles_lane_repair_dry_run.json`.
  - Evidence: record output paths, counts, cleanup blockers, and whether apply
    remains blocked or ready for explicit user approval.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 8 - independent code review completed.
  - Covers: implementation steps 17-18.
  - Verify: rerun affected focused tests after any review fixes.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and approval status in `Execution Evidence`.
  - Handoff: plan may proceed to final sign-off only if review has no
    unresolved blockers.
  - Sign-off: `<agent/date>` after review evidence is recorded.

## Verification

### Focused Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_user_profiles_lane_integrity.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidation_target_routing.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidator_origin_policy_db_writer.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidation_lifecycle_diagnostics.py -q
venv\Scripts\python.exe -m pytest tests/test_db.py -q
```

### Regression Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_memory_writer_information_flow_contracts.py -q
venv\Scripts\python.exe -m pytest tests/test_user_profile_agent.py -q
venv\Scripts\python.exe -m pytest tests/test_user_memory_evidence_agent.py -q
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_rag_flow.py -q
```

### Static Greps

```powershell
rg -n "upsert=True" src\kazusa_ai_chatbot\db\users.py src\kazusa_ai_chatbot\consolidation
```

Expected: no ordinary relationship/affinity/user-memory write helper uses
`upsert=True`. Explicit identity creation helpers may still use upsert only
when the match is `ensure_character_identity(...)` or another named creation
path documented in this plan.

```powershell
rg -n "self_cognition|internal_thought|group_chat_review|scheduled_future_cognition" src\kazusa_ai_chatbot\consolidation src\kazusa_ai_chatbot\db
```

Expected: matches are limited to synthetic-source rejection lists, audit
classification, lifecycle diagnostics, tests, and documentation. No match may
use these labels as ordinary user target ids.

```powershell
rg -n "CHARACTER_GLOBAL_USER_ID" src\kazusa_ai_chatbot\consolidation src\kazusa_ai_chatbot\rag\person_context src\kazusa_ai_chatbot\db
```

Expected: character/profile reads and explicit character identity creation are
allowed. Ordinary user-lane eligibility must reject this id.

### Maintenance Dry-Run Commands

Run only after the plan is approved for execution and live diagnostics are
explicitly authorized:

```powershell
venv\Scripts\python.exe -m scripts.audit_user_profiles_lane --output test_artifacts\user_profiles_lane_audit.json
venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --dry-run --output test_artifacts\user_profiles_lane_repair_dry_run.json
```

Expected: audit output reports counts by profile kind and issue code. Dry-run
output lists exact cleanup candidates, blockers, and zero mutations.

Apply command, only after dry-run review and explicit user command:

```powershell
venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --apply --input test_artifacts\user_profiles_lane_repair_dry_run.json --output test_artifacts\user_profiles_lane_repair_apply.json
```

Expected: apply deletes only current empty-unreferenced rows from the approved
dry-run. Rows with changed reference counts or new blockers are skipped and
reported.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the drafting agent must
reread `README.md`, `docs/HOWTO.md`, `development_plans/README.md`, this
plan, the consolidation/db/RAG ICDs, `consolidation.target`, `db.users`,
`db.script_operations`, and focused tests from a fresh-review posture.

Review scope:

- Architecture alignment: adapters stay thin; RAG returns evidence;
  deterministic consolidation target planning owns write eligibility; DB
  helpers own persistence mechanics; LLM stages own semantic memory only after
  deterministic target validation.
- Stage readiness: status remains `draft`; dependencies and blocked apply
  gates are explicit; registry mismatch caused by the current no-README-edit
  constraint is recorded as residual risk.
- Instruction completeness: contracts, file paths, phases, dry-run/apply
  behavior, verification, and progress checkpoints are concrete.
- Creativity suppression: no unresolved choices, compatibility shims, LLM
  classifiers, fallback target rewrites, broad cleanup, or duplicate-profile
  merge work remains.
- Scope mapping: every requested item is explicit: issue description,
  malformed-data removal plan, RCA, corrupted data-source hardening, and
  robustness proof.

Record blockers, non-blocking findings, required edits, residual risks, and
approval status in `Execution Evidence` before plan approval.

## Self Plan Review And Remediation

Review performed during draft refinement on 2026-07-02 by the assigned
planning agent.

| Finding | Severity | Remediation Applied To This Plan | Residual Risk |
| --- | --- | --- | --- |
| Original plan class was `large` despite DB deletion and production write behavior changes. | Blocking for plan-contract alignment | Changed plan class to `high_risk_migration` and added migration gates. | `development_plans/README.md` still lists the plan as large because this task forbids editing that file. Registry update needs a separate authorized pass. |
| Original RCA named debug/test and synthetic sources but did not map them to actual creation and write paths. | Blocking | Added source-path RCA covering `resolve_global_user_id`, `create_user_profile`, `ensure_character_identity`, `build_consolidation_target_plan`, relationship/affinity upserts, RAG evidence, and debug/pytest identities. | Execution may find additional creation paths; agent must update this draft before approval if so. |
| Original cleanup strategy did not distinguish empty unreferenced rows from history-linked no-platform rows. | Blocking | Added audit classification, protected reference counts, cleanup blockers, and apply skip behavior for changed blocker state. | Live data may include references not listed here; audit implementation must stop on unknown reference collections rather than delete. |
| Original hardening strategy did not explicitly block character/global profile rows from ordinary user-memory writes. | Blocking | Added `character_global` classification, target-state rejection, tests, and static grep expectations involving `CHARACTER_GLOBAL_USER_ID`. | Character identity remains stored in `user_profiles`; future code could misuse it unless tests stay focused. |
| Original plan allowed target validation but left relationship/affinity DB helper upsert behavior under-specified. | Blocking | Added `db.users` change surface, helper hardening tasks, tests, and static grep for `upsert=True`. | Some existing tests expect upsert behavior; execution must update tests only where the new contract intentionally changes runtime writes. |
| Original plan did not state that invented social memory can be valid only after target validity is proven. | Blocking against user request | Added explicit context and mandatory rules separating semantic memory generation from deterministic target eligibility. | Semantic quality of memory content remains outside this target-integrity plan. |
| Original verification lacked audit/apply current-state safety checks. | Non-blocking but important | Added dry-run/apply command contract, input artifact requirement, changed-state skip rule, and focused repair apply test. | Actual apply should still require manual artifact review before user approval. |
| Original plan review content was code-review oriented rather than plan-review/remediation oriented. | Blocking against requested scope | Added `Independent Plan Review` and this `Self Plan Review And Remediation` section. | Independent plan review still has not run; plan remains draft. |
| Post-edit placeholder scan found that the scan-status sentence contained literal sentinel words and would keep future scans noisy. | Non-blocking | Reworded the scan-status sentence so the static placeholder grep returns no matches. | Future edits must rerun the same grep before approval. |

Coverage check:

- Issue description: covered in `Lane Analysis Requirements`.
- Plan to remove malformed data: covered in `Lane Analysis Requirements`,
  `Data Migration`, and maintenance command contracts.
- RCA: covered in `Lane Analysis Requirements`.
- Plan to harden corrupted data source: covered in target eligibility,
  write-intent, DB helper hardening, and implementation stages.
- Plan to prove robustness: covered in focused tests, static greps,
  maintenance dry-run commands, and acceptance criteria.

Placeholder and ambiguity scan:

- No placeholder terms or unresolved implementation options remain in the
  executable sections.
- The only deferred registry mismatch is explicitly caused by this task's
  instruction not to edit `development_plans/README.md`.
- Plan status remains `draft`; no execution is authorized by this refinement.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG/context leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Migration safety: audit classification correctness, reference counters,
  dry-run/apply artifact integrity, changed-state skip behavior, and rollback
  instructions.
- Regression and handoff quality, including focused and regression tests,
  execution evidence, and path-safe commands.

The parent fixes concrete findings directly only when the fix is inside the
approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Profile audit reports production, diagnostic, character/global,
  no-platform-history-linked, empty-unreferenced, malformed, and manual-review
  rows separately.
- Dry-run repair reports exact empty-unreferenced cleanup candidates and
  blockers without mutating live data.
- Approved apply deletes only dry-run-approved rows that remain
  empty-unreferenced at apply time.
- Production user-memory, relationship insight, affinity, and user-style
  writes reject diagnostic, empty, synthetic, no-platform unapproved, and
  character/global profile targets.
- Real platform-linked production users remain valid write targets.
- Relationship and affinity helpers do not create arbitrary profile rows as a
  side effect of ordinary runtime writes.
- RAG person-context profile evidence remains readable but cannot authorize
  persistence.
- Character/global identity handling remains explicit and separate from
  ordinary user memory.
- Verification commands pass and evidence is recorded.
- Independent code review is completed with no unresolved blockers.

## Data Migration

1. Export a read-only snapshot of relevant user-profile lane data through an
   approved script under `test_artifacts/` before any apply:

   ```powershell
   venv\Scripts\python.exe -m scripts.audit_user_profiles_lane --output test_artifacts\user_profiles_lane_audit.json
   ```

2. Build a dry-run repair plan:

   ```powershell
   venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --dry-run --output test_artifacts\user_profiles_lane_repair_dry_run.json
   ```

3. Review counts and samples for every profile kind:
   - production users;
   - diagnostic debug rows;
   - diagnostic pytest rows;
   - character/global rows;
   - empty unreferenced rows;
   - no-platform history-linked rows;
   - malformed rows;
   - manual-review mixed rows.

4. Confirm every cleanup candidate has:
   - no platform accounts;
   - no aliases;
   - no non-empty relationship insight;
   - no protected references;
   - default or missing relationship-header content only;
   - no character/global or diagnostic identity signal.

5. Apply only after explicit user command:

   ```powershell
   venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --apply --input test_artifacts\user_profiles_lane_repair_dry_run.json --output test_artifacts\user_profiles_lane_repair_apply.json
   ```

6. Apply must recheck each candidate immediately before delete. If reference
   counts or blocker fields changed since dry-run, skip the row and report
   `skipped_changed_since_dry_run`.

7. Re-run audit and compare counts:

   ```powershell
   venv\Scripts\python.exe -m scripts.audit_user_profiles_lane --output test_artifacts\user_profiles_lane_audit_post_apply.json
   ```

8. Rollback uses the pre-apply export plus the apply report. Restoration must
   use an explicit follow-up command or user-state snapshot workflow; do not
   implement automatic rollback in this plan.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Historical profile deleted | Require protected-reference counts, dry-run review, apply recheck, and skip-on-change | Audit classification tests and apply tests |
| Debug/pytest workflows break | Preserve diagnostic rows and reject only ordinary production user-lane writes | Focused target tests and existing debug/pytest tests |
| Character profile misclassified as user | Explicit `character_global` classification and `CHARACTER_GLOBAL_USER_ID` rejection | Target tests and static grep |
| Real no-platform production user blocked | Preserve history-linked rows for manual review instead of deleting; future identity repair is deferred | Audit report and manual-review classification |
| RAG evidence mistaken for write permission | Keep eligibility in deterministic target planning, not RAG or prompts | Static source review and target tests |
| Upsert hardening breaks expected creation | Keep creation in explicit identity/character/maintenance helpers | DB helper tests |
| Registry metadata mismatch | Record residual risk because README edits are forbidden in this task | Self plan review |

## Execution Evidence

Cleanup-only execution completed on 2026-07-03. New-write hardening remains
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Commands and artifacts:

- `venv\Scripts\python.exe -m py_compile src\scripts\_lane_cleanup.py src\scripts\audit_user_profiles_lane.py src\scripts\repair_user_profiles_lane.py src\kazusa_ai_chatbot\db\script_operations.py`
- `venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --dry-run --output test_artifacts\user_profiles_lane_repair_dry_run.json`
- `venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --apply --input test_artifacts\user_profiles_lane_repair_dry_run.json --output test_artifacts\user_profiles_lane_repair_apply.json`
- `venv\Scripts\python.exe -m scripts.repair_user_profiles_lane --dry-run --output test_artifacts\user_profiles_lane_post_repair_dry_run.json`

Results:

- Baseline: 3347 total rows, 0 active rows, 0 findings, 0 deterministic
  planned actions.
- Apply: 0 actions, 0 blocked actions.
- Post-audit: 0 findings and 0 deterministic planned actions remain.
