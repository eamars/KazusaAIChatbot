# Modality Neutral Action Spec And Effector Expansion Execution Plan

## Summary

- Goal: implement the first runtime slice of the modality-neutral action spec:
  action contracts, shared validation, `send_message` compatibility, and
  character-selected lifecycle updates for `user_memory_units.active_commitment`.
- Plan class: high_risk_migration
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible
- Highest-risk areas: live cognition schema drift, action-ledger compatibility,
  memory lifecycle writes, and accidental side-channel execution
- Acceptance criteria: one shared action contract exists, `send_message`
  remains compatible, expired promises can be retired only through
  character-selected lifecycle action, and deferred tools have bounded handoff
  notes

This plan is approved for execution as of 2026-05-16. Execution must follow the
stage order, verification gates, and evidence requirements below.

## Context

Use
`development_plans/reference/designs/action_spec_effector_expansion_architecture.md`
as the design source for rationale, research context, decision records, and
future tool brainstorming.

This active plan is only the execution contract. Do not add design research,
new alternatives, or extra capability scope here during implementation.

## Mandatory Skills

Load these skills before execution:

- `development-plan-writing`: before changing this plan, registry rows, or
  execution evidence.
- `local-llm-architecture`: before changing cognition schema, prompts, graph
  state, action routing, evaluator behavior, or background LLM behavior.
- `no-prepost-user-input`: before changing promise, preference, commitment, or
  action persistence decisions.
- `py-style`: before editing Python files.
- `test-style-and-execution`: before adding, changing, or running tests.
- `cjk-safety`: before editing Python files containing CJK prompt text or CJK
  string literals.
  This applies in this plan only if Stage 3 edits Python prompt strings with
  CJK content; the English-only `action_spec` module still follows normal
  Python style.

## Mandatory Rules

- After any context compaction, reread this entire plan before continuing.
- After signing off a progress checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion or lifecycle status changes, run the `Independent Code
  Review` gate and record the result in `Execution Evidence`.
- Use `venv\Scripts\python` for Python commands.
- Do not read `.env` unless the user explicitly asks for environment
  inspection.
- Keep one shared semantic path:

```text
typed episode -> RAG/evidence -> cognition L1/L2/L3 -> dialog/private finalization
-> consolidation/persistence -> dispatcher/scheduler/effectors
```

- LLM stages own semantic judgment. Deterministic code owns schema validation,
  permissions, limits, execution eligibility, persistence, cache invalidation,
  scheduling, adapter delivery, and audit records.
- Do not remove, hide, or suppress promises based only on deterministic age, due
  date, keyword, or status.
- Do not add a new graph branch, prompt sidecar, fake user message, or separate
  cleanup channel for expired promises.
- Tool handlers must not call cognition directly and must not write final
  user-facing dialogue.
- Keep action specs optional in live cognition output until offline structured
  emission measurement is reviewed.
- Do not execute this plan until its status is explicitly changed to
  `approved` or `in_progress`.

## Must Do

- Execute this approved plan only through the stage order, verification gates,
  and evidence requirements below.
- Create `src/kazusa_ai_chatbot/action_spec/` with public action-spec contracts,
  capability registry, evaluator, attempt ledger, and memory lifecycle handler.
- Add `ActionSpecV1` with `cognition_mode`, `continuation`, source refs, target,
  params, urgency, visibility, deadline, and reason.
- Accept only `cognition_mode="deliberative"` in this plan. Reject `reflex`.
- Add `ActionSpecEvaluator` as the shared deterministic gate for action specs.
- Add prompt-safe capability projection for cognition.
- Reuse `self_cognition_action_attempts` through a generic action-attempt
  repository. Do not create a new action ledger collection.
- Preserve existing `send_message` behavior by bridging validated
  `ActionSpecV1(kind="send_message")` to existing `RawToolCall` and
  `TaskDispatcher.dispatch`.
- Add `memory_lifecycle_update` only for `user_memory_units.active_commitment`.
- Add a narrow `update_user_memory_unit_lifecycle` helper in
  `src/kazusa_ai_chatbot/db/user_memory_units.py`.
- Map lifecycle statuses:
  - `fulfilled -> completed`
  - `abandoned -> cancelled`
  - `obsolete -> archived`
  - `deferred -> active`
- Reject `EvolvingMemoryDoc` lifecycle targets without persistence.
- Run offline action-spec validity and latency measurement before enabling
  broad live `/chat` action-spec emission.
- Update the measured documentation surface before final sign-off. Documentation
  updates are part of the execution contract, not a cleanup task.

## Deferred

- Do not implement `schedule_self_check` runtime execution.
- Do not implement `web_research` or `fetch_url` runtime execution.
- Do not implement `note_open_loop` or `close_open_loop` runtime execution.
- Do not implement image generation.
- Do not change platform adapters.
- Do not change proactive-contact permission policy.
- Do not mutate `EvolvingMemoryDoc`.
- Do not rename, migrate, or delete MongoDB collections.
- Do not add arbitrary file, database, shell, HTTP, or external-message tools.
- Do not add deterministic cleanup jobs that retire promises by age.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Contracts reference | compatible prerequisite | Prerequisite satisfied on 2026-05-16; execute only through this approved plan's stage order and gates. |
| Action spec module | compatible additive | Add `action_spec` without removing dispatcher, self-cognition, or consolidator contracts. |
| `send_message` path | compatible bridge | Convert validated action specs to `RawToolCall` only at the bridge. Preserve `TaskDispatcher.dispatch`. |
| Action attempts ledger | compatible additive | Reuse `self_cognition_action_attempts`; add fields tolerantly. |
| `user_memory_units` lifecycle | compatible additive | Use existing statuses and one narrow lifecycle helper. No backfill. |
| Live cognition output | compatible gated | Keep action specs optional until offline measurement is reviewed. |
| Future tools | deferred | Do not ship runtime future tools under this plan. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- Compatibility surfaces are limited to the rows listed above.
- Any collection rename, data migration, physical deletion, or broad live-chat
  rollout requires a separate approved plan or explicit plan revision.
- If an existing helper cannot support lifecycle update safely, stop and update
  this plan instead of adding an ad hoc MongoDB write.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts and owner boundaries in this plan.
- The agent must not introduce alternate migration strategies, fallback paths,
  feature flags, extra tools, or broad compatibility layers.
- The agent must not implement deferred tools.
- Changes outside the listed `Change Surface` require explicit plan revision.
- If equivalent validation, projection, or ledger behavior exists, adapt it into
  the approved module boundary instead of duplicating it.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

After execution:

- Cognition output may include optional `ActionSpecV1` records.
- Existing final-dialog and `action_directives` behavior remains compatible.
- Validated `send_message` action specs use the existing dispatcher path.
- Validated `memory_lifecycle_update` action specs can update
  `user_memory_units.active_commitment` lifecycle state.
- Action attempts are recorded through the existing
  `self_cognition_action_attempts` backing collection.
- Deferred tools are documented but not runnable.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Action spec ownership | `ActionSpecV1` is semantic cognition residue, not a dispatcher tool call. | Keeps LLM semantic action choice separate from deterministic execution mechanics. |
| Dispatcher boundary | Bridge only validated `send_message` action specs into existing `RawToolCall` and `TaskDispatcher.dispatch`. | Preserves adapter-facing delivery, scheduler validation, permission checks, and duplicate suppression. |
| Memory lifecycle owner | Route `memory_lifecycle_update` to the user-memory owner, not dispatcher. | Promise retirement is private persistence, not adapter-facing delivery. |
| Promise retirement semantics | Let L2 choose `fulfilled`, `abandoned`, `obsolete`, or `deferred`; deterministic code maps only validated decisions to collection statuses. | Avoids age-based or keyword-based cleanup while keeping writes inspectable. |
| Action ledger | Back generic action attempts with `self_cognition_action_attempts`. | Reuses the existing idempotency collection and avoids a second control ledger. |
| Live rollout | Keep action specs optional and gated by offline validity and latency measurement. | Protects live chat from local-LLM structured-output drift. |
| Future tools | Reserve next-stage tools in documentation only. | Prevents this plan from shipping web research, notes, image generation, or scheduler cognition loops. |

## Contracts To Implement

Use `TypedDict`, `Literal`, dataclasses, and explicit validator functions for
internal action contracts. Do not introduce Pydantic for this internal module.

The implementation must mirror the approved
`development_plans/reference/designs/cognition_contracts_design.md` contract
names. The inline shapes below are the executable slice for this plan. If these
shapes conflict with the approved reference at implementation time, stop and
update the plan instead of silently choosing one.

### Supporting Types

```python
PolicyRefV1 = str


class EvidenceRefV1(TypedDict):
    schema_version: Literal["evidence_ref.v1"]
    evidence_kind: Literal[
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "system_event",
        "external_document",
    ]
    evidence_id: str
    owner: str
    excerpt: str | None
    observed_at: str | None


class ActionSourceRefV1(TypedDict):
    schema_version: Literal["action_source_ref.v1"]
    ref_kind: Literal[
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "cognitive_episode",
        "system_event",
    ]
    ref_id: str
    owner: str
    relationship: Literal["basis", "target", "lineage", "result", "audit"]
    evidence_refs: list[EvidenceRefV1]


class ActionTargetV1(TypedDict):
    schema_version: Literal["action_target.v1"]
    target_kind: Literal[
        "current_user",
        "current_channel",
        "user",
        "channel",
        "memory_unit",
        "cognitive_episode",
        "self",
        "none",
    ]
    target_id: str | None
    owner: str
    scope: dict[str, object]


class ActionContinuationV1(TypedDict):
    schema_version: Literal["action_continuation.v1"]
    mode: Literal[
        "none",
        "immediate_followup",
        "scheduled_followup",
        "background_followup",
    ]
    episode_type: str | None
    max_depth: int
    include_result_as: str | None


class CapabilitySpecV1(TypedDict):
    schema_version: Literal["capability_spec.v1"]
    capability_kind: str
    category: Literal["action"]
    owner_module: Literal["dispatcher", "memory_lifecycle", "orchestrator"]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    handler_id: str
    lifecycle_hooks: list[str]
    permission_policy: PolicyRefV1
    rate_limit_policy: PolicyRefV1
    audit_policy: PolicyRefV1
    prompt_projection_policy: PolicyRefV1
```

### `ActionSpecV1`

```python
class ActionSpecV1(TypedDict):
    schema_version: Literal["action_spec.v1"]
    kind: str
    cognition_mode: Literal["deliberative", "reflex"]
    source_refs: list[ActionSourceRefV1]
    target: ActionTargetV1
    params: dict[str, object]
    urgency: Literal["now", "background", "scheduled"]
    visibility: Literal["private", "preview", "user_visible"]
    deadline: str | None
    continuation: ActionContinuationV1
    reason: str
```

### `ActionEvalResult`

```python
class ActionEvalResult(TypedDict):
    ok: bool
    action_spec: ActionSpecV1 | None
    capability: CapabilitySpecV1 | None
    idempotency_key: str | None
    handler_owner: str | None
    errors: list[str]
```

### Initial Capability Params

`send_message` capability params:

```python
class SendMessageParamsV1(TypedDict):
    target_channel: Literal["same"] | str
    text: str
    execute_at: str | None
    delivery_mentions: list[dict[str, object]]
```

The evaluator must bridge validated `SendMessageParamsV1` to the existing
dispatcher `RawToolCall(tool="send_message", args=...)` shape without exposing
dispatcher handler IDs to cognition prompts.

`memory_lifecycle_update` capability params:

```python
class MemoryLifecycleUpdateParamsV1(TypedDict):
    memory_kind: Literal["user_memory_unit"]
    unit_type: Literal["active_commitment"]
    unit_id: str
    lifecycle_decision: Literal[
        "fulfilled",
        "abandoned",
        "obsolete",
        "deferred",
    ]
    due_at: str | None
```

The action target for `memory_lifecycle_update` must be:

```python
{
    "schema_version": "action_target.v1",
    "target_kind": "memory_unit",
    "target_id": unit_id,
    "owner": "user_memory_units",
    "scope": {"unit_type": "active_commitment"},
}
```

Prompt-safe capability projection must include only semantic capability name,
availability, visibility, allowed lifecycle decisions, and a short parameter
summary. It must not include `handler_id`, raw MongoDB collection names,
credentials, raw channel IDs, adapter IDs, or database internals.

### Lifecycle Decision Vocabulary

Stage 3 L2 prompt guidance must use this exact vocabulary:

| L2 value | Meaning | Collection status |
|---|---|---|
| `fulfilled` | The character judges the commitment has been satisfied or should be recorded as completed because the promised action was carried out. | `completed` |
| `abandoned` | The character explicitly decides not to pursue the commitment anymore because continuing it would be inappropriate, unwanted, or no longer character-consistent. | `cancelled` |
| `obsolete` | Newer context makes the commitment irrelevant or superseded without implying failure or refusal. | `archived` |
| `deferred` | The character decides the commitment is still valid and should remain open for later handling. | `active` |

L2 must choose among these values semantically and provide `reason`. Deterministic
code only validates source refs, target ownership, transition legality, and
persistence mechanics. `deferred` must not suppress retrieval of the commitment.

### `update_user_memory_unit_lifecycle`

```python
async def update_user_memory_unit_lifecycle(
    unit_id: str,
    *,
    status: Literal["active", "archived", "completed", "cancelled"],
    timestamp: str,
    reason: str,
    action_attempt_id: str,
    due_at: str | None = None,
) -> UserMemoryUnitDoc | None: ...
```

The helper updates only lifecycle fields, `updated_at`, and a `merge_history`
audit entry. It rejects missing units, non-`active_commitment` units, invalid
status transitions, and empty reasons.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/action_spec/__init__.py`
- `src/kazusa_ai_chatbot/action_spec/models.py`
- `src/kazusa_ai_chatbot/action_spec/registry.py`
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
- `src/kazusa_ai_chatbot/action_spec/attempt_ledger.py`
- `src/kazusa_ai_chatbot/action_spec/handlers/memory_lifecycle.py`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `tests/test_action_spec_models.py`
- `tests/test_action_spec_evaluator.py`
- `tests/test_action_spec_attempt_ledger.py`
- `tests/test_action_spec_memory_lifecycle.py`
- `tests/test_action_spec_self_cognition_bridge.py`
- `tests/fixtures/action_spec_cognition_cases.json`
- `src/scripts/measure_action_spec_validity.py`

`tests/fixtures/` already exists for JSON case data in this repo. This plan
uses it only for non-Python offline measurement fixtures. Python test modules
remain flat under `tests/`.

### Modify

- `development_plans/README.md`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/db/self_cognition.py`
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
- `src/kazusa_ai_chatbot/self_cognition/handoff.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/db/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/dispatcher/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- Conditional documentation files listed in `Documentation Change Surface
  Measurement` when implementation touches their contracts.

The `persona_supervisor2_cognition_l3.py` change is limited to L3/L4 pass-through
of optional action specs if the current collector would otherwise drop them.
L3 must not turn action specs into text, change lifecycle decisions, or choose
execution owners.

### Keep

- `src/kazusa_ai_chatbot/dispatcher/` remains the owner of adapter-facing
  tools. Only bridge code needed for `ActionSpecV1 -> RawToolCall` is allowed.
- `scheduled_events` schema remains unchanged.
- `EvolvingMemoryDoc` repository remains unchanged.
- Platform adapters remain unchanged.

### Delete

- No code, collection, or document is deleted in this plan.

## Documentation Change Surface Measurement

Measured on 2026-05-16 from the approved contract reference, this plan's code
change surface, and existing subsystem READMEs.

### Direct Documentation Updates

These files must be updated before Stage 6 can pass:

| Document | Why It Becomes Stale | Required Update |
|---|---|---|
| `src/kazusa_ai_chatbot/action_spec/README.md` | New public module. | Document package boundary, `ActionSpecV1`, `CapabilitySpecV1` action entries, evaluator, attempt ledger, dispatcher bridge, memory lifecycle handler, continuation policy, deferred tools, and forbidden paths. |
| `src/kazusa_ai_chatbot/nodes/README.md` | Cognition output can include optional action specs while preserving existing `action_directives`. | Document that L2 may emit modality-neutral action specs, L3/dialog still own wording, L3 visual/image path is separate from action execution, and action specs remain optional/gated. |
| `src/kazusa_ai_chatbot/dispatcher/README.md` | `send_message` can arrive through an action-spec bridge before becoming `RawToolCall`. | Document that dispatcher remains execution owner for adapter-facing tools, not the parent action system; explain `ActionSpecV1(kind="send_message") -> RawToolCall -> TaskDispatcher.dispatch`. |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | Self-cognition handoff and duplicate state route through the generic action-attempt contract. | Document action-spec validation before `send_message` handoff, private `memory_lifecycle_update` for commitment retirement, and tolerant reuse of `self_cognition_action_attempts`. |
| `src/kazusa_ai_chatbot/db/README.md` | `user_memory_units.active_commitment` gains a lifecycle helper and `self_cognition_action_attempts` gains generic action-attempt metadata. | Document allowed lifecycle transitions, no age-based cleanup, action-attempt compatibility, and existing-status filtering for retired commitments. |

### Conditional Documentation Audit

Audit these files during Stage 6. Update them only if implementation touches the
named boundary; otherwise record "no doc change required" with grep evidence:

| Document | Update Trigger |
|---|---|
| `src/kazusa_ai_chatbot/event_logging/README.md` | Any new event family, recorder, payload field, forbidden-data rule, or action-spec validation telemetry is added. |
| `src/kazusa_ai_chatbot/proactive_output/README.md` | Any proactive-contact permission, preview, outbox, or external-message boundary changes. |
| `src/kazusa_ai_chatbot/brain_service/README.md` | Any service startup, runtime adapter registration, scheduler callback, or delivery handoff behavior changes. |
| `src/kazusa_ai_chatbot/memory_evolution/README.md` | Any `EvolvingMemoryDoc` lifecycle mutation becomes supported. This plan should leave it unchanged. |
| `docs/HOWTO.md` | Any operator-facing command, config flag, API behavior, or runbook step is added beyond the developer-only measurement script. |
| root `README.md` | Public architecture table or package map becomes inaccurate after implementation. |

### Documentation Staleness Checks

Stage 6 must run static documentation greps and inspect each match:

```powershell
rg -n "only.*send_message|send_message.*only|currently send_message|Action candidates always use|dispatch_shape: \"send_message\"|future effector|one-effector" README.md docs src development_plans -g "*.md"
rg -n "ActionSpecV1|CapabilitySpecV1|memory_lifecycle_update|self_cognition_action_attempts|active_commitment|TaskDispatcher|RawToolCall" src\kazusa_ai_chatbot\action_spec\README.md src\kazusa_ai_chatbot\nodes\README.md src\kazusa_ai_chatbot\dispatcher\README.md src\kazusa_ai_chatbot\self_cognition\README.md src\kazusa_ai_chatbot\db\README.md
```

Matches are not automatically failures. They must either describe the new
boundary accurately or be updated in the same stage.

## Overdesign Guardrail

- Actual problem: the character can notice an expired promise, but the system
  lacks a shared action contract for retiring that promise without speaking or
  deterministic age cleanup.
- Minimal change: add action contracts, shared evaluator, existing
  `send_message` bridge, and one private lifecycle capability for
  `user_memory_units.active_commitment`.
- Ownership boundaries: cognition chooses semantic action; evaluator validates;
  dispatcher owns adapter-facing delivery; user-memory repository owns
  commitment lifecycle writes; event logging owns sanitized observability.
- Rejected complexity: no arbitrary tools, no new ledger collection, no
  `EvolvingMemoryDoc` mutation, no schedule-self-check runtime tool, no web
  research runtime tool, no reflex execution, no adapter changes, no cleanup
  job.
- Evidence threshold: add any rejected capability only through a reviewed
  follow-up plan that names owner, handler, continuation policy, latency budget,
  tests, and permission gates.

## Implementation Order

### Stage 0: Contracts Reference Reconciliation

- Verify `development_plans/reference/designs/cognition_contracts_design.md`
  is approved and linked from the registry.
- Verify the reference still resolves the prior approval blockers:
  `reflection_promoted` is not a committed trigger source, `system_event` is
  reserved rather than committed, and `ActionSourceRefV1`, `ActionTargetV1`,
  `ActionContinuationV1`, and `CapabilitySpecV1` are defined.
- Verify this plan and
  `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`
  remain reconciled against contracts 3, 4, and 7.
- Verify this plan's registry row still marks it approved before implementation
  starts.

Exit criteria:

- The contracts reference approval is verified.
- The blocker-resolution check above is recorded with grep output or explicit
  section references.
- Reconciliation notes are present in `Independent Plan Review` or
  `Execution Evidence`.

### Stage 1: Tests For Action Contracts

- Add tests for `ActionSpecV1` schema validation.
- Add tests for supporting shapes: `ActionSourceRefV1`, `ActionTargetV1`,
  `ActionContinuationV1`, and action-category `CapabilitySpecV1`.
- Add tests for capability registry projection, including exact
  `memory_lifecycle_update` params schema and lifecycle vocabulary.
- Add anti-leakage assertions for `send_message` and
  `memory_lifecycle_update` prompt-safe projection: no `handler_id`, no raw
  collection name, no credentials, no adapter IDs, no raw channel IDs, and no
  database internals.
- Add tests that `reflex` is rejected for all current capabilities.
- Add tests for continuation policy validation.
- Add tests for unsupported `EvolvingMemoryDoc` lifecycle targets.

Expected pre-implementation result:

- Focused tests fail with missing module or missing symbol errors.

### Stage 2: Implement `action_spec` Module

- Implement `models.py`.
- Implement `registry.py`.
- Implement `evaluator.py`.
- Implement `attempt_ledger.py`.
- Export public symbols from `__init__.py`.
- Back the ledger with `self_cognition_action_attempts`.

Exit criteria:

- Stage 1 tests pass.
- Old `self_cognition_action_attempts` rows remain readable.

### Stage 3: Cognition Schema And Offline Measurement

- Add optional action-spec state fields.
- Extend cognition output contract validation.
- Update L2 guidance for deliberative action-spec decisions using the exact
  `Lifecycle Decision Vocabulary` in this plan.
- Update L3/L4 collection only as needed to pass through optional action specs
  unchanged while preserving existing `action_directives` behavior. L3 must not
  render, strip, or reinterpret action specs.
- Preserve existing `action_directives` behavior in L3 collector output.
- Add offline fixture cases.
- Add `src/scripts/measure_action_spec_validity.py`.
- Run the offline measurement.

Exit criteria:

- Existing cognition schema tests pass.
- Measurement records JSON validity, action-spec validity, false-positive action
  emission, invalid continuation requests, prompt size impact, and latency
  impact.
- Broad live `/chat` emission remains disabled until the measurement is reviewed.

### Stage 4: `send_message` Compatibility Bridge

- Route existing self-cognition `send_message` candidates through
  `ActionSpecV1` validation.
- Convert validated `send_message` action specs to existing `RawToolCall`.
- Dispatch through existing `TaskDispatcher.dispatch`.
- Preserve delivery mentions, permission checks, scheduler validation, adapter
  availability, and duplicate suppression.
- Persist attempts through the generic action-attempt repository.
- Add an automated end-to-end self-cognition fixture covering:
  self-cognition source case -> `ActionSpecV1(kind="send_message")`
  validation -> `RawToolCall(tool="send_message")` bridge ->
  `TaskDispatcher.dispatch`, with unchanged scheduled-event and duplicate
  suppression behavior.

Exit criteria:

- Existing dispatcher and self-cognition send-message tests pass.
- Existing action-attempt duplicate suppression remains effective.
- The self-cognition action-spec bridge fixture passes and proves no direct
  adapter call, no bypassed dispatcher validation, and no changed delivery
  mention metadata.

### Stage 5: Memory Lifecycle Capability

- Add `update_user_memory_unit_lifecycle`.
- Implement `handlers/memory_lifecycle.py`.
- Validate source refs and cognition-authored reason.
- Map semantic lifecycle statuses to collection-native statuses.
- Write lifecycle audit metadata to the action-attempt ledger and
  `merge_history`.
- Ensure active-promise retrieval excludes retired statuses through the existing
  active-status filter.
- Reject `EvolvingMemoryDoc` targets without persistence.

Exit criteria:

- A character-selected lifecycle action can mark a long-past-due active
  commitment as `completed`, `cancelled`, or `archived`.
- No deterministic age threshold hides or changes promises.
- Unsupported memory-evolution targets are rejected without writes.

### Stage 6: Documentation And Handoff

- Update every file in `Direct Documentation Updates`.
- Audit every file in `Conditional Documentation Audit`.
- Document the action-spec owner boundary.
- Document that dispatcher is an execution owner, not the parent action system.
- Document how `ActionSpecV1(kind="send_message")` bridges to `RawToolCall`.
- Document how `memory_lifecycle_update` changes
  `user_memory_units.active_commitment` status without age-based cleanup.
- Document compatibility expectations for old and new
  `self_cognition_action_attempts` rows.
- Add next-stage handoff notes for `schedule_self_check`, `web_research`, and
  notes/open-loop tools.
- Run the documentation staleness checks and inspect all matches.

Exit criteria:

- Direct documentation files describe the accepted action flow, owner
  boundaries, lifecycle behavior, and deferred tools.
- Conditional documentation files are either updated or explicitly recorded as
  not requiring a change.
- Static greps do not leave stale documentation claiming `send_message` is the
  only future effector without context.
- Execution evidence records changed doc paths, audited no-change doc paths,
  and grep output summaries.

## Progress Checklist

- [ ] Stage 0 - contracts reference prerequisite complete
  - Verify: registry links approved `cognition_contracts_design.md`.
  - Evidence: record reference path and reconciliation notes.
  - Sign-off: `<agent/date>`
- [ ] Stage 1 - action contract tests added
  - Verify: focused tests fail for missing symbols before implementation.
  - Evidence: record failing commands.
  - Sign-off: `<agent/date>`
- [ ] Stage 2 - `action_spec` module implemented
  - Verify: action-spec model, evaluator, registry, and ledger tests pass.
  - Evidence: record changed files and test output.
  - Sign-off: `<agent/date>`
- [ ] Stage 3 - cognition schema and offline measurement gated
  - Verify: cognition schema tests pass and measurement output is reviewed.
  - Evidence: record validity, false-positive, and latency results.
  - Sign-off: `<agent/date>`
- [ ] Stage 4 - `send_message` bridge complete
  - Verify: dispatcher and self-cognition regression tests pass.
  - Evidence: record duplicate-suppression and old-row readability result.
  - Sign-off: `<agent/date>`
- [ ] Stage 5 - memory lifecycle capability complete
  - Verify: memory lifecycle and user-memory retrieval tests pass.
  - Evidence: record status mapping and unsupported-target rejection.
  - Sign-off: `<agent/date>`
- [ ] Stage 6 - documentation and handoff complete
  - Verify: direct documentation files are updated; conditional documentation
    audit is recorded; documentation static greps pass after inspection.
  - Evidence: record changed doc paths, no-change audited doc paths, and grep
    output summaries.
  - Sign-off: `<agent/date>`
- [ ] Independent code review complete
  - Verify: review findings are fixed or recorded as residual risks; affected
    tests are rerun.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, and
    approval status.
  - Sign-off: `<agent/date>`

## LLM And Latency Budget

- Do not add an extra live LLM call to ordinary user-message turns.
- Keep action specs optional in cognition output.
- Cap emitted action specs to 3 per cognition episode.
- Cap prompt-visible affordances to registered capabilities relevant to the
  episode source.
- Tool handlers return `ToolResult` artifacts only; orchestrator owns any
  continuation scheduling in future plans.
- Offline measurement must record prompt-size and latency impact before broad
  live `/chat` action-spec emission is enabled.

## Verification

Run these checks after implementation:

```powershell
venv\Scripts\python -m pytest tests/test_action_spec_models.py
venv\Scripts\python -m pytest tests/test_action_spec_evaluator.py
venv\Scripts\python -m pytest tests/test_action_spec_attempt_ledger.py
venv\Scripts\python -m pytest tests/test_action_spec_memory_lifecycle.py
venv\Scripts\python -m pytest tests/test_action_spec_self_cognition_bridge.py
venv\Scripts\python -m pytest tests/test_dispatcher.py tests/test_self_cognition_integration.py
venv\Scripts\python -m pytest tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_persona_supervisor2_schema.py
venv\Scripts\python -m pytest tests/test_user_memory_units_rag_flow.py tests/test_db.py
venv\Scripts\python src\scripts\measure_action_spec_validity.py --fixture tests\fixtures\action_spec_cognition_cases.json
rg -n "only.*send_message|send_message.*only|currently send_message|Action candidates always use|dispatch_shape: \"send_message\"|future effector|one-effector" README.md docs src development_plans -g "*.md"
rg -n "ActionSpecV1|CapabilitySpecV1|memory_lifecycle_update|self_cognition_action_attempts|active_commitment|TaskDispatcher|RawToolCall" src\kazusa_ai_chatbot\action_spec\README.md src\kazusa_ai_chatbot\nodes\README.md src\kazusa_ai_chatbot\dispatcher\README.md src\kazusa_ai_chatbot\self_cognition\README.md src\kazusa_ai_chatbot\db\README.md
```

Targeted behavioral checks after deterministic tests pass:

- long-past-due promise, character fulfills;
- long-past-due promise, character abandons with reason;
- long-past-due promise, character defers without lifecycle action;
- unrelated live user message, no lifecycle action.

## Independent Code Review

Run this gate after all verification commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan and inspect the full diff from
a fresh-review posture.

Review scope:

- Mandatory skill compliance.
- Plan alignment for `Must Do`, `Deferred`, `Change Surface`, and
  implementation stages.
- Code quality and ownership boundaries.
- Absence of hidden fallback paths, prompt leaks, direct adapter calls, direct
  cognition calls from handlers, generic MongoDB writes, or deterministic
  semantic promise cleanup.
- Verification evidence and next-stage handoff quality.

Fix findings only inside the approved change surface. If a finding requires new
scope, update this plan or request approval before changing code.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Action spec becomes a second dispatcher | Keep dispatcher as execution owner only for adapter-facing tools | Bridge tests and owner-boundary review |
| Promise lifecycle becomes deterministic cleanup | Require validated character-selected action spec with source refs and reason | No-action and past-due fixtures |
| Ledger compatibility breaks duplicate suppression | Back action attempts with `self_cognition_action_attempts` and read old rows tolerantly | Ledger and self-cognition regression tests |
| Live cognition schema increases invalid output | Keep action specs optional and run offline measurement first | Measurement script and cognition tests |
| Memory writes bypass repository owner | Add narrow lifecycle helper and reject generic writes | Memory lifecycle tests |
| Documentation keeps stale single-effector or dispatcher-parent claims | Treat direct documentation updates and stale-string greps as Stage 6 exit criteria | Documentation change surface evidence |
| Prompt-safe affordance projection leaks runtime internals | Assert projection excludes handler IDs, raw IDs, credentials, adapter IDs, and DB internals | Capability projection anti-leakage tests |

## Acceptance Criteria

- `cognition_contracts_design.md` is approved and reconciliation is recorded.
- `ActionSpecV1` and `ActionSpecEvaluator` exist in `action_spec`.
- Supporting contracts `ActionSourceRefV1`, `ActionTargetV1`,
  `ActionContinuationV1`, and action-category `CapabilitySpecV1` are
  implemented and tested.
- `reflex` is represented in schema but rejected for all current capabilities.
- `memory_lifecycle_update` capability params and lifecycle vocabulary match
  this plan exactly.
- Prompt-safe capability projection anti-leakage tests pass.
- `send_message` works through the action-spec bridge without behavior
  regression.
- Automated self-cognition bridge fixture proves
  self-cognition -> action spec -> dispatcher behavior.
- `memory_lifecycle_update` can mark active commitments completed, cancelled,
  or archived only after a validated character-selected action.
- No deterministic age threshold removes or hides active promises.
- Action attempts use the existing `self_cognition_action_attempts` backing
  collection.
- `EvolvingMemoryDoc` lifecycle targets are rejected without persistence.
- Offline structured-emission measurement is reviewed before broad live `/chat`
  action-spec emission.
- Direct documentation files from `Documentation Change Surface Measurement`
  are updated.
- Conditional documentation audit is recorded with either changes or explicit
  no-change evidence.
- Static documentation grep matches are inspected and stale claims are fixed.
- Next-stage handoff notes are updated.

## Independent Plan Review

Run this gate before approval, execution, or handoff.

Review scope:

- Architecture doc and this execution plan are separated cleanly.
- This plan contains execution instructions only, not design research or
  open-ended brainstorming.
- The contracts-reference prerequisite is satisfied, and this plan status is
  approved before execution starts.
- Change surface, tests, verification, progress checklist, and acceptance
  criteria are specific enough for implementation.
- Deferred tools are not executable under this plan.

Latest review result:

- 2026-05-15: contracts design draft created after prior approval review.
  - Status change: `cognition_contracts_design.md` now exists as a draft
    reference.
  - Remaining blocker: the contracts design is not yet approved and this plan
    has not been reconciled against it.
  - Decision: this plan remains `draft` and must not be executed.
- 2026-05-15: independent plan review requested for approval.
  - Inputs reviewed: development plan registry, active execution plan, action
    spec architecture reference, and presence check for
    `development_plans/reference/designs/cognition_contracts_design.md`.
  - Blocker at review time:
    `development_plans/reference/designs/cognition_contracts_design.md` was
    absent. The execution plan's summary, mandatory rules, cutover policy,
    implementation order, and acceptance criteria required this reference before
    approval.
  - Decision: approval denied. The plan remains `draft` and must not be
    executed.
  - Current follow-up after contracts draft creation: approve
    `cognition_contracts_design.md`, then record reconciliation against this
    plan and the architecture reference.
- 2026-05-15: active-agent fresh review completed after architecture/execution
  split; no separate reviewer was available in this session.
  - Historical blocker at review time:
    `development_plans/reference/designs/cognition_contracts_design.md` did not
    exist, so this plan stayed draft.
  - Non-blocking finding: architecture research, decision justification, and
    future tool brainstorming now live in
    `development_plans/reference/designs/action_spec_effector_expansion_architecture.md`.
  - Non-blocking finding: the active plan now contains execution scope, change
    surface, stages, progress checklist, verification, review gates, and
    acceptance criteria without open-ended design alternatives.
  - Approval status: not approved for execution; ready for
    `cognition_contracts_design.md` prerequisite work.
- 2026-05-16: contracts-reference prerequisite resolved.
  - Inputs reviewed: approved
    `development_plans/reference/designs/cognition_contracts_design.md`,
    action-spec architecture reference, and this execution plan.
  - Reconciliation recorded: `ActionSpecV1` references now use
    `ActionSourceRefV1`, `ActionTargetV1`, and `ActionContinuationV1`;
    evaluator capability references now use action-category `CapabilitySpecV1`
    entries.
  - Remaining status: this plan is still `draft` and is not approved for
    execution until a final independent plan review explicitly changes its
    status.
- 2026-05-16: documentation change surface measured and added to this plan.
  - Inputs reviewed: current change surface, subsystem README list, nodes,
    dispatcher, self-cognition, DB, event logging, proactive-output, and
    brain-service documentation.
  - Direct documentation updates now required for `action_spec`, cognition
    nodes, dispatcher, self-cognition, and DB docs.
  - Conditional documentation audit now required for event logging, proactive
    output, brain service, memory evolution, HOWTO, and root README when their
    boundaries are touched.
  - Stage 6 now requires static staleness greps and recorded evidence.
- 2026-05-16: execution-plan review blockers addressed.
  - Inputs reviewed: approved contracts reference, this plan's
    `Contracts To Implement`, implementation stages, verification, and
    documentation gates.
  - Contracts dependency check: current contracts reference is approved,
    removes `reflection_promoted` from committed triggers, reserves
    `system_event`, and defines `ActionSourceRefV1`, `ActionTargetV1`,
    `ActionContinuationV1`, `MemoryQueryV1`, `MemoryResultV1`,
    `MemoryLifecycleUpdateV1`, and `CapabilitySpecV1`.
  - Plan update: supporting TypedDicts, capability params, lifecycle decision
    vocabulary, prompt-safe projection anti-leakage assertions, L3 pass-through
    boundary, self-cognition bridge fixture, CJK-safety applicability, and
    `tests/fixtures/` justification are now explicit.
  - Remaining status: this plan is still `draft` until final independent plan
    review approves execution.
- 2026-05-16: final independent plan review completed for approval.
  - Inputs reviewed: plan-writing contract, execution gates, cutover policy,
    development plan registry, approved contracts reference, action-spec
    architecture reference, and this execution plan.
  - Approval blockers: none remaining. Prior contracts-reference blockers are
    resolved in the approved reference and verified in this plan's Stage 0.
  - Non-blocking finding resolved before approval: added the required
    `Design Decisions` section so the executable plan satisfies final-plan
    contract structure.
  - Decision: approved for execution. Execute only through the stage order,
    verification commands, documentation gates, and independent code review
    gate in this plan.

## Execution Evidence

No implementation has been executed. This is an approved execution plan.
