# Cross-User Character Memory Scope And Authority Bugfix Plan

- Status: superseded
- Superseded on: 2026-08-24
- Superseded by:
  `asuna_semantic_authority_and_memory_feedback_consolidated_bugfix_plan_2026-08-24.md`
- Lifecycle note: valid RCA, architecture, audit, and database-approval evidence
  is carried forward by the consolidated plan. This record grants no current
  execution or database authority.
- Date: 2026-08-23
- Type: active bugfix plan
- RCA evidence:
  `test_artifacts/diagnostics/asuna_group_480386272_memory_personality_policy_rca_20260823.md`
- RCA evidence SHA-256:
  `681A4C3113D0C0E5ED3F43E7D897D449931DCC70005D811D6E2F3A8112EB2285`
- Production implementation authority: requires a later explicit user command.
- Database mutation authority: requires a separate user-approved exact apply
  manifest after the read-only audit.

## Summary

The observed QQ group behavior is not the intended product behavior. The
platform and participant identifiers are distinct, but target-specific
relationship, consent, promise, and intimate-role material can be promoted to
global character memory and later retrieved with factual authority. A public
group scene then makes the other participant visible without making the first
participant's agreement transferable. The system currently preserves the
identity boundary mechanically and loses it semantically.

The repair keeps Kazusa's hybrid agentic design:

1. LLM stages continue to decide semantic meaning, including whether an
   accepted behavior is genuinely global, user-scoped, group-scoped,
   temporary, or character-identity evidence.
2. Deterministic code validates the LLM's structured scope certificate,
   preserves provenance, maps typed memory lanes to cognition authority, and
   rejects invalid writes.
3. The live cognition path remains exactly one `A1 -> A2 -> G -> P` pass with
   no added foreground LLM call, retry, database read, agent, or tool call.
4. Existing group public-scene and current-participant continuity packets stay
   separate. A concise recipient-applicability policy is added to existing
   cognition guidance instead of creating another packet or arbiter.
5. Existing polluted memory is audited and remediated by exact memory-unit ID.
   Valid global memories remain; invalid units are rejected through the
   existing lifecycle and cache invalidation path. Whole-database deletion is
   not part of this design.

The only recurring model-cost increase is at most one batched background
review call during a daily reflection promotion that actually proposes a
memory write. Consolidation already has a specialist and reviewer, so its call
count is unchanged.

## Problem Statement And Root Cause

The confirmed failure chain is:

```text
target-specific conversation
  -> global self-guidance or global fact candidate
  -> persistence without an explicit recipient-scope certificate
  -> shared-memory retrieval
  -> projection loses memory type and source authority
  -> cognition receives the row as direct character/world fact
  -> public group context exposes a different participant
  -> the prior participant's agreement is treated as generally applicable
```

The failure is not QQ identifier collapse. It is a combined write-authority,
retrieval-authority, and recipient-applicability defect:

- `consolidation/character_self_guidance.py` asks for global behavior in prose
  but does not require a structured global-applicability decision. Persisted
  rows hardcode low privacy risk and an empty source user.
- `reflection_cycle/promotion.py` can promote relationship-derived material to
  global fact or self-guidance without an independent target-removal review,
  and source evidence currently enters with a hardcoded low privacy risk.
- `persona_supervisor2_rag_projection.py` can aggregate unlike memory types
  into one formatted row and discard the typed metadata needed to distinguish
  fact from conditional self-guidance.
- `persona_supervisor2_cognition.py` maps every non-user-continuity memory row
  to direct `character_world_context`, even though promoted self-guidance is
  already correctly treated as conditional when it arrives through the
  separate promoted-reflection path.
- Cognition already distinguishes public scene from current participant
  continuity, but it has no compact general rule stating that public
  observability does not transfer a named participant's consent, permission,
  promise, relationship, or intimate role to another participant.

### Additional read-only audit seed: exchange-condition promotion

The 2026-08-24 semantic-progression replay found one active target-free
promoted row that belongs in this plan's read-only audit:

- memory unit ID: `reflection_e0c29b7a8e2be3bff65b64640a719495`;
- memory name: `交换条件互动模式`;
- `memory_type`: `fact`;
- `source_kind`: `reflection_inferred`;
- `authority`: `reflection_promoted`;
- top-level source user: empty;
- evidence: many reflection runs across four scope references.

Its original content is:

```text
在与用户或群聊成员进行信息查询、任务协助等实用性互动时，习惯于将单纯的信息提供转化为带有角色特色的“交换条件”仪式（例如：索要奖励、要求对方付出某种代价作为交换），以此维持互动的趣味性与权力动态。
```

This row is causal evidence of global character pressure, not evidence that the
fresh replay identity inherited another user's raw profile or commitment. Its
many cross-scope evidence refs also mean its disposition cannot be assumed from
one awkward response. The read-only audit must decide whether it is valid
target-free character lore, conditional behavioral self-guidance, unsupported
self-reinforcement, or another typed disposition under this plan's independent
review contract.

No database action is authorized by adding this audit seed. If the reviewed
result proposes rejection or another lifecycle action, the exact memory unit ID
must appear in the separately user-approved apply manifest.

## Historical Design Review

### Decisions To Carry Forward

| Historical plan | Design retained by this plan |
| --- | --- |
| `cognition_v3_hybrid_agentic_loop_reconciliation_plan.md` | Preserve bounded semantic stages, deterministic mechanics, matching-evidence role binding, and a compact prompt. Memory evidence is not persona or final stance. |
| `cognition_v3_handleless_model_contract_bigbang_plan_2026-08-22.md` | Keep exactly one A1, A2, G, and P call. LLM stages own semantic judgment; deterministic code owns provenance, validation, limits, and persistence. |
| `cognition_subjective_continuity_dialog_quality_plan_2026-08-23.md` | Preserve the five authority lanes. Participant continuity proves prior actors, actions, and outcomes only. Conditional character context cannot establish current facts, consent, commitments, permissions, capabilities, or user intent. |
| `consolidator_lane_router_memory_pollution_bigbang_plan.md` | Preserve target planning, coarse LLM routing, lane specialist, lane reviewer, deterministic source/write validation, and the distinction among current-user commitment, global character guidance, group norm, temporary roleplay, and identity growth. |
| `cognition_core_v2_character_identity_growth_bigbang_plan.md` | Character identity remains global only through target-free abstraction. User relationships, private promises, named roles, exact utterances, and intimate details remain scoped. Identity/boundary growth stays owned by the separate identity pipeline. |
| `memory_evidence_scoped_user_continuity_plan.md` | Preserve exact current-user filtering and provenance fields. User continuity never becomes shared fact. No new response-path model call or database read. |
| `qq_group_public_scene_response_ordering_bugfix_plan.md` | Preserve visible public speaker order and addresses while keeping current-participant continuity separate. The existing participant-branch isolation requirement remains authoritative. |
| `group_topic_continuity_authority_fix_plan.md` | Public scene may establish what was said and to whom. Self-guidance remains a conditional tactic rather than current permission or fact. Keep one primary response objective and grounded character judgment. |
| `cognition_v3_consolidation_interaction_subtext_handoff_bugfix_plan_2026-08-23.md` | Preserve the fixed execution arrangement: one GPT-5.6 Luna executor at maximum reasoning and standard normal speed owns production implementation and test execution; an independent owner performs sign-off. |

### Historical Mechanisms Not To Reintroduce

- No Cognition V2 fallback, goal bids, scalar `W`, foreground multi-agent
  debate, sibling-candidate salvage, semantic scorer, semantic rewrite, or
  semantic regeneration loop.
- No deterministic keyword classifier over user prose, case-specific intimate
  vocabulary blocklist, post-LLM semantic rerouting, or response suppressor.
- No forced silence or blanket refusal. Asuna may independently accept or
  reject a different participant based on current context, personality,
  relationship, and boundaries; she may not inherit permission from somebody
  else's agreement.
- No persisted group transcript packet, group lock, extra ordering mechanism,
  compatibility alias, dual reader, feature flag, or fallback mapper.
- No second character-identity ledger and no path allowing self-guidance to
  rewrite identity or boundaries.
- No raw reflection output in live cognition and no routine live-chat lookup of
  the complete memory database.

## Confirmed Architecture Decisions

### 1. Write-Time Scope Certification

The existing consolidation router remains coarse. Its prompt receives one
general decision rule:

> Remove the source user, addressee, relationship target, and private scene.
> If the accepted behavior is no longer accurate and appropriate for the
> character with people generally, it is not global character self-guidance.

The rule is expressed generically and without the incident's exact prose.
Routing remains LLM-owned:

- a current-user agreement routes to `active_commitment`;
- a genuinely character-general accepted behavior routes to
  `character_self_guidance`;
- a group-specific convention routes to `interaction_style_image`;
- temporary roleplay writes nothing;
- a change to self-concept, personality judgment, or boundary routes to
  `character_identity_growth`.

The existing self-guidance specialist and reviewer each independently return:

```json
{
  "global_applicability": "global | scoped | absent",
  "target_specific_meaning_removed": true,
  "affects_identity_or_boundaries": false,
  "private_detail_risk": "low | medium | high",
  "user_details_removed": true,
  "reason": "bounded semantic justification"
}
```

The specialist still proposes the candidate. The reviewer sees the original
authorized evidence as well as the candidate and makes its own scope decision;
it does not merely approve the specialist's label. Reviewer wording cleanup is
allowed only after the reviewer independently finds the candidate global.
Redaction alone cannot convert a scoped promise or relationship into global
guidance.

Deterministic persistence admits the write only when all of the following are
structurally true:

- specialist and reviewer both return `global_applicability=global`;
- both return `target_specific_meaning_removed=true`;
- both return `affects_identity_or_boundaries=false`;
- the final reviewer reports `private_detail_risk=low` and
  `user_details_removed=true`;
- the candidate has all required content and authorized source references;
- the output passes the canonical JSON parser and exact typed contract.

Any disagreement or invalid contract produces no self-guidance write. Code
does not infer a different semantic lane and does not alter the candidate's
meaning. The source conversation remains available to other already-owned
lanes selected by the router.

`MemoryPrivacyReview` is extended with the three scope fields above. New
`conversation_accepted` and `reflection_promoted` learned-character writes must
contain them. Seed and curated world knowledge keep their existing ownership
contracts.

### 2. Reflection Promotion Scope Review

The daily promotion prompt uses the same counterfactual target-removal rule and
must distinguish:

- target-free character or world fact;
- target-free conditional self-guidance;
- user-, relationship-, or group-scoped material;
- identity/boundary evidence owned by character identity growth.

Hourly and daily evidence cards carry their actual privacy notes or risk
labels. Missing source privacy assessment is represented as unreviewed rather
than silently converted to low risk.

When the promoter emits one or two write candidates, one independent reviewer
receives them in a single background call. It may accept or reject exact
candidates and may not rewrite them. A write requires promoter/reviewer
agreement on global applicability, target removal, and privacy. A candidate
with `affects_identity_or_boundaries=true` never writes through the memory
promotion path; the existing identity-growth process evaluates the same
eligible reflection evidence independently.

The reviewer has one attempt. Structural failure or disagreement fails closed
for that promotion run and leaves an auditable skipped disposition. This adds
at most one background LLM call per daily promotion with write candidates and
adds no live-chat latency.

### 3. Typed Retrieval Authority

Shared memory is partitioned before formatting. Rows with different
`memory_type`, `scope_type`, `source_kind`, or authority class are never merged
into one evidence entry. Projection preserves, at minimum:

- `memory_type`;
- `scope_type` and `scope_global_user_id` where applicable;
- `source_kind` and `source_system`;
- storage `authority`, `truth_status`, and `origin`;
- scope/privacy review when present;
- exact repository-owned source references for trace only.

The cognition connector maps typed metadata, never memory prose:

| Stored evidence | Cognition authority lane |
| --- | --- |
| exact current-user `user_continuity` | `participant_continuity` |
| `defense_rule` / character self-guidance | `conditional_character_guidance` |
| validated global `fact` or curated lore | `character_world_context` / direct facts |
| mixed, unsupported, or missing required learned-memory metadata | excluded from semantic evidence and recorded diagnostically |

This closes both the first-cycle shared-memory prewarm and normal RAG3 path.
The local resolver already preserves the relevant source metadata; the
projection and cognition connector must stop discarding it. Self-guidance can
shape tactics and boundaries but cannot prove that a current participant has
consented, promised, granted permission, accepted a role, or expressed a
current intention.

### 4. Recipient Applicability In Existing Cognition

A short general policy is added to the existing A2, G, and P guidance:

> A public group scene proves what was said, by whom, and to whom. A
> relationship, promise, consent, permission, or role addressed to one
> participant does not transfer to another participant merely because both
> participants can see the scene.

A1 remains observation-focused. No output field or stage packet changes.
A2 evaluates participant scope, G chooses one character-grounded objective,
and P produces an action consistent with that scope. Conditional character
guidance can still influence style, willingness, or caution, while current
observation and the current participant's own relationship evidence determine
whether a new request is acceptable.

### 5. Existing Data Remediation

Database remediation is targeted and lifecycle-preserving:

1. Export active learned global `fact` and `defense_rule` rows from
   `conversation_accepted` and `reflection_promoted` sources with exact IDs,
   content hashes, origin, authority, source references, and privacy metadata.
2. Run a read-only offline LLM scope audit using the same target-removal,
   identity, and privacy contract. The script emits raw JSON only.
3. Produce a human-readable review that separates:
   - valid target-free global fact;
   - valid target-free conditional self-guidance;
   - scoped/private/relationship material;
   - identity-or-boundary material;
   - unresolved rows requiring human disposition.
4. Present the exact proposed apply manifest to the user. The manifest contains
   memory-unit IDs, expected content hashes, current statuses, reasons, and
   planned dispositions.
5. After separate explicit approval, back up the exact rows and re-fetch them.
   Stop on drift. Reject only approved invalid rows through
   `memory_evolution.reject_memory_unit(...)`, preserving audit history and
   invalidating cache.
6. Retain valid rows unchanged. Do not automatically rewrite, redact, reroute,
   or re-home historical content during this first repair.
7. Re-run the audit and prove that no approved-invalid row is active and that
   valid global positive controls still retrieve with the intended authority.

The code cutover and approved data remediation form one deployment gate. The
service must not reopen on the repaired code while known invalid active rows
remain. Rejected rows are not automatically reactivated during rollback.

## Target State

```text
current event + public group scene + exact current-participant continuity
                              |
                              v
RAG / first-cycle prewarm --typed memory projection--+
  facts ------------------------------> direct facts |
  self-guidance ------------> conditional context   |
  user continuity ----------> participant continuity|
                              |                      |
                              +----------+-----------+
                                         v
                                  A1 -> A2 -> G -> P
                                   one call per stage

post-turn consolidation                 daily reflection
  coarse router                          promoter
       |                                    |
  lane specialist                        candidate(s)
       |                                    |
  independent lane reviewer       one batched scope reviewer
       |                                    |
  deterministic scope/privacy gates and canonical persistence
```

## Runtime And Overhead Constraints

- Foreground cognition remains four calls: A1, A2, G, P.
- Foreground database reads, resolver calls, retries, agents, and tool calls do
  not increase.
- The public-scene and participant-continuity context budgets do not increase.
- Static cognition guidance grows only by the compact recipient rule.
- Consolidation call count is unchanged because its specialist and reviewer
  already exist.
- Reflection adds at most one batched call per daily run and only when a write
  candidate exists.
- Audit and remediation run offline and do not become runtime middleware.
- No new database collection, index, cache layer, or persisted group packet is
  introduced.

## Change Surface

### Production Files To Modify

- `src/kazusa_ai_chatbot/consolidation/lane_router.py`
  - add the generic scope distinction to the existing coarse semantic routing
    contract;
  - keep the lane list and router output shape unchanged.
- `src/kazusa_ai_chatbot/consolidation/character_self_guidance.py`
  - require independent specialist/reviewer scope judgments;
  - persist the actual reviewer scope/privacy certificate;
  - remove hardcoded low-risk and implicit-global decisions.
- `src/kazusa_ai_chatbot/memory_evolution/models.py`
  - extend the learned-memory privacy review contract with the exact scope
    fields.
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
  - preserve actual source privacy evidence;
  - add the batched independent scope reviewer and fail-closed write gate;
  - block identity/boundary candidates from the memory lane.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
  - partition shared memory by typed authority before aggregation and preserve
    source metadata.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - map typed memory evidence to direct, participant, or conditional authority.
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
  - add the compact recipient-applicability rule to A2, G, and P.
- `src/kazusa_ai_chatbot/db/script_operations.py`
  - add only bounded public read helpers needed to obtain and re-fetch exact
    migration rows if existing public operations cannot satisfy the audit.
- Relevant subsystem `README.md` files
  - document the canonical write certificate, retrieval mapping, recipient
    boundary, and migration lifecycle.

### Maintenance Tools To Add

- `src/scripts/audit_character_memory_scope.py`
  - read-only export and offline LLM classification;
  - raw JSON artifact output only;
  - no database mutation.
- `src/scripts/repair_character_memory_scope.py`
  - consume an exact approved manifest;
  - create a backup, verify hashes/status, reject exact IDs through the public
    lifecycle API, stop on drift, and verify cache invalidation.

If the current public database operations already provide every required exact
read, `db/script_operations.py` stays unchanged. That is the only conditional
file in the production change surface.

### Tests And Governance To Modify Or Add

- `tests/test_consolidation_lane_router_contract.py`
- `tests/test_consolidation_lane_bigbang_integration.py`
- `tests/test_consolidation_character_self_guidance_scope.py` (new)
- `tests/test_consolidation_memory_write_use_cases_live_llm.py`
- `tests/test_reflection_cycle_stage1c_promotion.py`
- `tests/test_rag_projection.py`
- `tests/test_shared_memory_prewarm.py`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py`
- `tests/unit/cognition_core_v3/test_prompt_context.py`
- `tests/test_character_memory_scope_migration.py` (new)
- `tests/test_script_db_boundary.py`
- `tests/ownership/source_test_impact_manifest.json`
- new raw and readable artifacts under `test_artifacts/live_llm/` and
  `test_artifacts/reviews/`.

The executor must patch only the exact affected rows in the ownership manifest
and preserve concurrent edits.

## Test Impact And Traceability

| Contract | Exact deterministic owner test | Regression prevented |
| --- | --- | --- |
| coarse global/user/group/transient distinction | `tests/test_consolidation_lane_router_contract.py::test_lane_router_distinguishes_global_user_group_and_transient_behavior_rules` | A user-scoped agreement is sent to global self-guidance before specialist review. |
| two independent global-scope decisions | `tests/test_consolidation_character_self_guidance_scope.py::test_self_guidance_write_requires_independent_global_scope_agreement` | One optimistic LLM decision persists a target-specific rule globally. |
| persisted review is real, not hardcoded | `tests/test_consolidation_character_self_guidance_scope.py::test_self_guidance_persists_reviewer_scope_and_privacy_review` | Learned rows falsely claim low privacy and removed user details. |
| reflection scope review | `tests/test_reflection_cycle_stage1c_promotion.py::test_reflection_memory_write_requires_independent_global_scope_review` | Relationship-derived reflection becomes global memory after one semantic pass. |
| identity ownership | `tests/test_reflection_cycle_stage1c_promotion.py::test_identity_or_boundary_candidate_never_writes_self_guidance` | Identity or boundary change bypasses character identity growth. |
| source privacy propagation | `tests/test_reflection_cycle_stage1c_promotion.py::test_promotion_evidence_carries_source_privacy_notes_without_assuming_low_risk` | Missing or risky source privacy is silently represented as low risk. |
| typed shared-memory projection | `tests/test_rag_projection.py::test_shared_memory_projection_partitions_fact_and_self_guidance_with_typed_metadata` | Fact and self-guidance collapse into one untyped direct-fact entry. |
| prewarm typed authority | `tests/test_shared_memory_prewarm.py::test_first_cycle_prewarm_preserves_self_guidance_conditional_authority` | First-cycle prewarm bypasses normal memory authority. |
| cognition connector mapping | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_rag_memory_authority_maps_self_guidance_to_conditional_context` | Any global memory type becomes direct character/world fact. |
| recipient applicability guidance | `tests/unit/cognition_core_v3/test_prompt_context.py::test_recipient_scoped_permission_rule_reaches_a2_goal_and_plan` | A public statement to participant A is treated as permission from participant B. |
| read-only audit | `tests/test_character_memory_scope_migration.py::test_audit_is_read_only_and_emits_exact_scope_manifest` | Diagnostic classification mutates production data or omits exact lineage. |
| drift-safe apply and cache invalidation | `tests/test_character_memory_scope_migration.py::test_apply_rejects_only_approved_unchanged_units_and_invalidates_cache` | Repair affects unapproved rows or leaves rejected units cached. |
| apply fail-closed on drift | `tests/test_character_memory_scope_migration.py::test_apply_stops_on_manifest_row_drift` | Approval is applied to content different from the reviewed row. |
| database ownership boundary | `tests/test_script_db_boundary.py::test_scripts_do_not_import_raw_or_private_db_boundary` | Maintenance scripts bypass the public repository lifecycle. |

Assertions test typed contracts, target identity, permission authority, and
write disposition. They do not assert the incident's exact wording or use a
keyword blacklist as a proxy for semantics.

## Required Live LLM Evaluation

Run live tests individually and inspect raw input, raw model output, parsed
contract, retrieved evidence, authority lanes, persistence decision, and final
visible response after each case. Produce a human-readable review artifact.

1. Reproduce the captured group pattern: one participant establishes a
   target-specific service or intimate contract; a different participant asks
   for access. IDs remain distinct and Asuna does not cite the first contract
   as permission. She may make a separate character-grounded decision.
2. A private wife/master/slave/service agreement routes to user scope or no
   global write, according to the character's acceptance and the owning lane.
3. Positive control: `以后你也可以偶尔用“收到”回应大家。` remains eligible for
   global self-guidance and can affect later interactions with another user.
4. A close relationship experience may produce a target-free identity-growth
   abstraction while the relationship, promise, and intimate details remain
   scoped.
5. One-turn roleplay writes nothing durable.
6. A group norm remains group-scoped.
7. Mixed fact and self-guidance retrieval reaches separate direct and
   conditional authority lanes.
8. Reflection promotion rejects target-specific relationship material while
   retaining a genuinely target-free global positive control.

The captured production slice is used as a failure-pattern input, not as a
literal prompt rule. Each real-LLM test follows the repository's one-case-at-a-
time execution and review contract.

## Execution Roles

### Architecture And Acceptance Owner

- Owner: parent architect.
- Responsibilities: preserve the system boundary, adjudicate scope conflicts,
  review the complete diff and evidence, and decide plan closure.
- Authority: read-only architecture and independent acceptance until the user
  explicitly approves implementation.
- Database gate: present the exact remediation manifest and obtain separate
  user approval before any apply operation.

### Implementation And Verification Owner

- Fixed executor: exactly one GPT-5.6 Luna subagent.
- Reasoning effort: maximum.
- Runtime speed: standard normal speed.
- Responsibilities: production implementation, deterministic tests, live LLM
  tests, migration tooling, scoped documentation, and evidence artifacts.
- Constraint: the executor owns the files named by the approved plan, records
  their pre-handoff status and hashes, preserves concurrent changes, and does
  not spawn further agents.
- Gate: starts only after an explicit implementation command and plan approval.
- Database apply: excluded until the user separately approves an exact apply
  manifest.

### Independent Sign-Off Owner

- Must be independent of the Luna implementation executor.
- Reviews the scoped diff, exact tests, raw/live artifacts, overhead claims,
  database backup and drift checks, service shutdown, and residual risks.
- Returns explicit finding dispositions before lifecycle closure.

## Mandatory Skills And Rules For Execution

- `development-plan` for lifecycle, gates, evidence, and closeout.
- `local-llm-architecture` for the smallest canonical LLM/deterministic
  ownership contract.
- `no-prepost-user-input` to prevent deterministic classification or rewriting
  of user intent.
- `py-style` before any Python change.
- `cjk-safety` for Python files containing Chinese or Japanese text.
- `test-style-and-execution` before changing or running tests.
- `character-test` for live behavior evaluation and inspected run artifacts.
- `llm-trace-debug` for protected prompt, stage, and output evidence.
- The parent architecture owner authors the human-readable live evaluation.
- `database-data-pull` for read-only diagnostic export.
- Use `venv\Scripts\python` for Python and pytest commands.
- Use the canonical JSON parser and bounded typed contract handling.
- Preserve user and concurrent work; use `apply_patch` for manual edits.
- Read no `.env` as part of implementation or testing. Normal service startup
  may load its configured runtime environment through existing code.

## Cutover And Rollback

This is a big-bang contract correction with no dual semantics:

1. Approve the plan and record the owned baseline.
2. Implement code, deterministic tests, live tests, and read-only audit tooling.
3. Pass focused and adjacent deterministic verification.
4. Pass the required live cases one at a time.
5. Run the read-only production-memory audit and review every proposed invalid
   row.
6. Obtain separate user approval for the exact apply manifest.
7. Back up and reject approved invalid rows, prove cache invalidation, and
   re-audit.
8. Start the service on the canonical contract and run the captured cross-user
   smoke case plus positive controls.
9. Complete independent sign-off and archive the plan only when every gate is
   closed.

Code rollback uses the last known-good code revision only after preserving all
audit and migration artifacts. Rejected rows remain rejected; automatic data
reactivation is prohibited because it would reintroduce known-invalid semantic
state. Any proposed reactivation requires a separate reviewed manifest.

## Acceptance Criteria

- QQ/public participant IDs remain exact and distinct throughout the captured
  scenario.
- User A's consent, permission, promise, relationship, or intimate role is not
  treated as User B's authority merely because both appear in a public group
  scene.
- Character-global self-guidance still influences behavior conditionally and
  does not establish current facts or permissions.
- Genuinely global positive controls continue to consolidate, retrieve, and
  influence later interactions.
- User-specific commitments, group norms, temporary roleplay, and identity
  growth reach their existing canonical owners.
- Fact and self-guidance rows retain typed metadata and enter different
  cognition authority lanes in both prewarm and RAG3 paths.
- A1/A2/G/P call count, foreground database reads, foreground retries, and
  live response topology are unchanged.
- Consolidation call count is unchanged; reflection adds no more than one
  candidate-gated daily background call.
- Read-only audit proves its zero-write behavior.
- Approved remediation rejects only exact, unchanged invalid rows, preserves a
  backup and lifecycle history, invalidates cache, and retains valid memories.
- All focused deterministic, adjacent, live LLM, migration, manifest, and
  independent-review gates pass with inspectable evidence.

## Deferred And Explicitly Out Of Scope

- Whole-database erasure or broad memory reset.
- General retuning of Asuna's personality, relationship intensity, response
  ratio, engagement threshold, or dialog style.
- QQ adapter changes, participant-ID migration, or group-history redesign.
- A new general memory ontology, vector-store migration, or retrieval engine.
- New foreground agents, tools, retries, or semantic arbitration stages.
- Automatic historical-memory rewriting or re-homing.
- Changes to unrelated ongoing agentic-resolver work.

## Agent Autonomy Boundaries

The implementation owner may choose local helper names, bounded prompt wording,
test fixture values, artifact directory names, and command order within the
approved contracts. The owner may omit `db/script_operations.py` when existing
public operations fully satisfy the exact audit and refetch requirements.

Any need to change the lane set, add a foreground model call, add a database
collection or index, introduce deterministic prose classification, alter the
QQ adapter contract, rewrite existing memory content, broaden the database
apply set, or modify a file outside the approved change surface is a plan
conflict. The executor returns that conflict to the architecture owner and
waits for explicit direction.

## Progress Checklist

- [x] Reproduced and documented the behavior against the latest anchored
  24-hour QQ group slice.
- [x] Confirmed participant identifiers remain distinct.
- [x] Traced the write, retrieval, cognition-authority, and group-recipient
  boundaries.
- [x] Reconciled the solution with completed group, cognition, consolidation,
  memory, identity-growth, and reflection plans.
- [x] Fixed the proposed architecture, overhead ceiling, execution roles, and
  source/test boundary.
- [ ] User approves this plan and explicitly commands production
  implementation.
- [ ] Luna executor records the baseline and completes the scoped code and
  deterministic verification.
- [ ] Luna executor completes one-at-a-time live LLM evaluation and artifacts.
- [ ] Read-only memory audit and human review are complete.
- [ ] User approves the exact database apply manifest.
- [ ] Approved targeted remediation and post-apply verification are complete.
- [ ] Independent sign-off has no unresolved required finding.
- [ ] Plan is archived and the lifecycle registry is updated to completed.
