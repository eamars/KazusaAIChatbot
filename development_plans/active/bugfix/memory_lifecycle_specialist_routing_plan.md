# memory lifecycle specialist routing plan

## Summary

- Goal: make `memory_lifecycle_update` a router-level intent selected by L2d,
  and move active-commitment lifecycle judgment, target selection, and
  executable update construction into a dedicated memory lifecycle specialist.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `cjk-safety`
- Overall cutover strategy: bigbang internal action-kind cutover; no
  compatibility shim for the old L2d-owned executable memory update path
- Highest-risk areas: live response path latency, L2d/action-spec ownership
  boundaries, active-commitment target binding, DB lifecycle execution,
  self-cognition private lifecycle closure, and content-anchor handoff for
  stale promise avoidance
- Capacity policy: review at most 12 active commitment aliases per specialist
  call; materialize at most 3 executable lifecycle updates per user turn;
  overflow remains active and is recorded in trace/log evidence for a later
  turn.
- Acceptance criteria: L2d can select `memory_lifecycle_update` without
  choosing a commitment target; the specialist can bind one of many active
  commitments by alias and emit executable `apply_memory_lifecycle_update`;
  L2d never materializes DB `unit_id` lifecycle specs; fulfilled tiramisu
  commitments stop resurfacing as active obligations in downstream context

## Context

The current L2d implementation in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py` treats
`memory_lifecycle_update` as a directly executable memory action. It exposes
the capability only when `_select_active_commitment(state)` finds exactly one
active commitment, and `_build_memory_lifecycle_action_spec(...)` then binds
that one target into a DB-owned action spec.

That boundary fails when a user has multiple active commitments. The original
QQ private-chat failure showed exactly this case: the user delivered the
tiramisu, but multiple active dessert/magic commitments were present. L2d could
not deterministically select one target, so the stale tiramisu row stayed
active even after fulfillment evidence was stored elsewhere. Later cognition
kept seeing the active obligation and reopened “提拉米苏还没兑现”.

The accepted design principle is the same as `speak`: L2d decides whether the
capability is needed, not the domain-specific content. For `speak`, L2d
selects a text surface and L3/dialog decide what to say. For memory lifecycle,
L2d must select the need for memory lifecycle review, and a specialist must
decide which commitment changed state.

Experiment evidence exists in
`experiments/l2d_commitment_alias_experiment.py` and trace
`test_artifacts/llm_traces/l2d_commitment_alias_experiment__20260517T140754819496Z.json`.
The experiment proved the specialist-style prompt can select
`commitment_1 -> fulfilled` for the stale tiramisu row, while the existing L2d
node selects only `speak`.

Capacity validation was added to the same experiment and run as a single live
case:

```powershell
venv\Scripts\python.exe experiments\l2d_commitment_alias_experiment.py capacity_12_commitments_last_alias_fulfilled
```

Trace:
`test_artifacts/llm_traces/l2d_commitment_alias_experiment__20260517T200937403540Z.json`.
The specialist prompt received 12 aliases from `commitment_1` through
`commitment_12`, selected `commitment_12 -> fulfilled`, and materialized the
trusted fixture update for `capacity-12-blueberry-cheesecake` with no validation
errors. The existing L2d path still emitted the known warning that it dropped a
lifecycle request without a target commitment, reinforcing the need for this
plan's ownership split.

## Mandatory Skills

- `development-plan-writing`: load before changing this plan, the registry,
  execution evidence, or lifecycle status.
- `local-llm-architecture`: load before changing L2d, specialist prompts,
  graph routing, action-spec capability boundaries, RAG/cognition payloads, or
  live response-path LLM calls.
- `no-prepost-user-input`: load before changing any commitment interpretation,
  promise acceptance, lifecycle decision, or persistence channel.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK prompt text.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Execute this plan only after its registry status is changed to `approved` or
  `in_progress`.
- L2d owns only semantic capability routing: `speak`,
  `memory_lifecycle_update`, and `trigger_future_cognition`.
- L2d must not choose active-commitment handles, DB IDs, lifecycle target
  fields, collection names, repository payloads, or lifecycle status updates.
- The memory lifecycle specialist owns semantic lifecycle judgment for active
  commitments. It may choose visible aliases such as `commitment_1`; it must
  never see or emit `unit_id`.
- Deterministic code owns alias creation, alias validation, alias-to-`unit_id`
  resolution, action-spec validation, DB persistence, audit records, graph
  routing, and execution ordering.
- Do not add deterministic keyword matching over user text to decide
  fulfillment. If model output is wrong, fix the prompt, schema, or tests.
- Do not add retry loops, fallback prompts, compatibility action kinds,
  side-channel DB writes, or broad memory refactors.
- Real LLM tests must be run one at a time and inspected one at a time.
- Prompt-facing payloads must remain semantic and bounded; raw storage IDs and
  internal collection names must not enter L2d or specialist prompts.

## Must Do

- Keep `memory_lifecycle_update` as the L2d-selected router intent name.
- Rename the current executable DB lifecycle action kind to
  `apply_memory_lifecycle_update`.
- Add a dedicated memory lifecycle specialist stage that receives current
  input, L2 judgment, active commitment aliases, retrieved evidence, and
  conversation progress, then emits lifecycle decisions by alias plus
  downstream content-anchor roles.
- Add deterministic alias binding and validation for all active commitments in
  the current user memory context up to the 12-alias specialist cap, not only a
  single selected commitment.
- Run the memory lifecycle specialist after L2d selects
  `memory_lifecycle_update` and before selected L3 text/dialog execution, so
  content anchors can avoid reopening fulfilled commitments.
- Ensure private-only memory lifecycle updates still execute and consolidate
  when L2d does not select `speak`.
- Update action-spec registry, evaluator, execution, handler, tests, and docs
  so `memory_lifecycle_update` is a route intent and
  `apply_memory_lifecycle_update` is the executable DB update.
- Update self-cognition lifecycle paths to use the new specialist boundary or
  a deterministic direct route that produces `apply_memory_lifecycle_update`
  only after an explicit target is already owned by self-cognition.
- Add deterministic, patched LLM, integration, and live LLM tests listed in
  this plan.

## Deferred

- Do not redesign active-commitment extraction or memory-unit consolidation.
- Do not add a new MongoDB collection or migrate existing memory rows.
- Do not change the public `/chat` response shape.
- Do not introduce a generic tool framework for all memory updates.
- Do not make L3 or dialog decide whether memory should update.
- Do not add broad content-anchor redesign beyond consuming the specialist's
  prompt-safe role anchors.
- Do not solve unrelated stale commitments outside active commitments.
- Do not implement multi-turn repair for ambiguous lifecycle decisions.

## Cutover Policy

Overall strategy: bigbang internal cutover.

| Area | Policy | Instruction |
|---|---|---|
| L2d `memory_lifecycle_update` meaning | bigbang | Replace direct executable lifecycle binding with a router intent. Do not preserve the old single-target materializer. |
| Executable DB lifecycle action | bigbang | Rename the executable action kind to `apply_memory_lifecycle_update` and update evaluator, execution, handler, tests, and docs. |
| Specialist LLM | bigbang | Add the specialist as the only live-chat owner of active-commitment target selection. |
| Existing memory documents | compatible | Preserve existing `user_memory_units` data shape and lifecycle statuses. No data migration. |
| Public service response | compatible | Preserve `/chat` output shape and adapter behavior. |
| Tests and docs | bigbang | Rewrite tests/docs that claim L2d owns executable lifecycle target binding. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not keep the old executable `memory_lifecycle_update` DB path
  as a fallback.
- Any lingering executable DB action using kind `memory_lifecycle_update` is a
  plan violation.
- Any change to this cutover policy requires user approval before
  implementation.

## Overdesign Guardrail

- Actual problem: fulfilled active commitments can remain active when multiple
  commitments exist because L2d currently requires a single deterministic
  target before it can materialize a lifecycle update.
- Minimal change: keep L2d as a capability router, add one specialist for
  active-commitment lifecycle decisions, and rename the executable DB action
  to `apply_memory_lifecycle_update`.
- Ownership boundaries: L2d selects the memory lifecycle route; the specialist
  chooses semantic lifecycle decisions by alias; deterministic code validates
  aliases and executes DB writes; L3/dialog consume prompt-safe role anchors.
- Rejected complexity: no compatibility shim for the old action kind, no
  keyword-based fulfillment detector, no retry loop, no new scheduler, no
  memory schema migration, no generic tool registry, no multi-agent debate, no
  platform-specific routing logic.
- Evidence threshold: add complexity only after a focused test or live trace
  proves the specialist cannot handle a supported active-commitment case with
  one bounded prompt and deterministic validation.

## Agent Autonomy Boundaries

- The agent may choose local helper names only when they preserve the public
  contracts in this plan.
- The agent must not introduce alternate lifecycle routes, compatibility
  action kinds, hidden fallback execution, or prompt repair loops.
- The agent must treat changes outside the files listed in Change Surface as
  high-scrutiny changes and record the justification before editing.
- The agent may remove old single-target L2d lifecycle code because that
  removal is explicitly in scope.
- If an existing helper already performs the needed validation or projection,
  reuse or move it instead of duplicating behavior.
- If the plan and code disagree, preserve the plan's ownership boundary and
  report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

The target runtime flow is:

```text
RAG retrieves active commitments
  -> L2d selects semantic actions:
       speak?
       memory_lifecycle_update?
       trigger_future_cognition?
  -> memory lifecycle specialist runs when selected
       aliases active commitments
       chooses target_alias + lifecycle decision
       emits prompt-safe content-anchor roles
       materializes apply_memory_lifecycle_update specs
  -> selected text surface runs when speak was selected
       consumes specialist role anchors
       avoids stale fulfilled-promise reopening
  -> action execution applies private DB lifecycle updates
  -> consolidation records action results and trace
```

`memory_lifecycle_update` remains the L2d intent name. It means “ask the memory
lifecycle specialist to review active commitments in this episode.” It does not
contain a `unit_id`.

`apply_memory_lifecycle_update` becomes the executable action kind. It means
“apply this already-resolved active-commitment lifecycle update to
`user_memory_units`.” It contains trusted `unit_id` fields created only by
deterministic code after specialist alias validation.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Router intent name | Keep `memory_lifecycle_update` | The user wants the visible intent to keep this name, matching `speak` as a selected semantic capability. |
| Executable action name | Use `apply_memory_lifecycle_update` | The verb `apply` clearly marks DB execution and separates it from route selection. |
| Specialist name | Use “memory lifecycle specialist” | It clearly names the owner for active-commitment lifecycle judgment while keeping L2d as a router. |
| Prompt target binding | Use deterministic aliases `commitment_1`, `commitment_2`, ... | Aliases are prompt-safe and avoid exposing raw storage IDs. |
| Active commitment review cap | Present at most 12 active commitment aliases to the specialist per call | This is enough for the observed failure class while keeping a local LLM prompt bounded and auditable. |
| Executable update cap | Materialize at most 3 lifecycle updates per user turn | A single user message can fulfill several promises, but batch-closing many memories in one response path is too much blast radius. |
| L2d target binding | Remove it | L2d is not the domain specialist and cannot safely choose among multiple active commitments. |
| Response-path order | Specialist before selected L3 text | L3/content anchors need the lifecycle result to avoid stale promise reopening. |
| DB execution | Keep existing repository owner | `user_memory_units.update_user_memory_unit_lifecycle` already owns persistence and audit merge history. |
| Missing no-change list | Treat omitted lifecycle list as empty only when the specialist explicitly outputs `no_lifecycle_change` | This is structural sanitation, not semantic reinterpretation. |

## Contracts And Data Shapes

### L2d router action

`memory_lifecycle_update` action specs selected by L2d must use this shape:

```python
{
    "schema_version": "action_spec.v1",
    "kind": "memory_lifecycle_update",
    "cognition_mode": "deliberative",
    "source_refs": [...],
    "target": {
        "schema_version": "action_target.v1",
        "target_kind": "cognitive_episode",
        "target_id": None,
        "owner": "memory_lifecycle_specialist",
        "scope": {"unit_type": "active_commitment"},
    },
    "params": {
        "review_kind": "active_commitment_lifecycle",
        "detail": "semantic reason for specialist review",
    },
    "urgency": "background",
    "visibility": "private",
    "deadline": None,
    "continuation": {"mode": "none", ...},
    "reason": "why L2d routed this review",
}
```

L2d must not emit `unit_id`, `target_alias`, `lifecycle_decision`, `due_at`, or
repository parameters for this router action.

### Specialist prompt payload

The specialist receives only prompt-safe active commitment rows:

```python
{
    "current_input": str,
    "formed_decision": {
        "logical_stance": str,
        "character_intent": str,
        "judgment_note": str,
        "internal_monologue": str,
    },
    "active_commitments": [
        {
            "target_alias": "commitment_1",
            "fact": str,
            "status": "active",
            "due_at": str | None,
            "due_state": str | None,
            "evidence_summary": str,
        }
    ],
    "memory_evidence": list[dict],
    "conversation_progress": dict | None,
}
```

The input builder must sort active commitments deterministically before aliasing
and pass no more than 12 rows. Omitted active commitments remain active and are
recorded in trace/log metadata, not hidden inside the prompt. The specialist
prompt must state that returned lifecycle decisions are priority-ordered from
clearest to weakest evidence.

### Specialist output

The specialist returns:

```python
{
    "decision": "lifecycle_change | no_lifecycle_change",
    "lifecycle_decisions": [
        {
            "target_alias": "commitment_1",
            "decision": "fulfilled | abandoned | obsolete | deferred",
            "role": "semantic role for downstream",
            "evidence_anchor": "short evidence string",
        }
    ],
    "content_anchor_roles": [
        {
            "role": "avoid_reopening | acknowledge_fulfillment | keep_waiting",
            "anchor": "prompt-safe downstream semantic anchor",
        }
    ],
}
```

For `no_lifecycle_change`, omitted `lifecycle_decisions` is normalized to an
empty list. Any lifecycle decision with an unknown alias, unsupported enum,
missing role, or missing evidence anchor is rejected and logged.

If the specialist returns more than 3 valid lifecycle decisions, deterministic
validation keeps the first 3 priority-ordered decisions and drops the surplus
with a warning. This enforces a response-path safety limit without changing the
semantic meaning of the kept decisions.

### Prompt-safe state handoff

The specialist writes this prompt-safe state key for L3/content-anchor
consumers:

```python
{
    "memory_lifecycle_context": {
        "schema_version": "memory_lifecycle_context.v1",
        "source": "memory_lifecycle_specialist",
        "decision": "lifecycle_change | no_lifecycle_change | skipped",
        "visible_alias_count": int,
        "omitted_alias_count": int,
        "lifecycle_decisions": [
            {
                "target_alias": "commitment_1",
                "decision": "fulfilled | abandoned | obsolete | deferred",
                "role": "semantic role for downstream",
                "evidence_anchor": "short evidence string",
            }
        ],
        "content_anchor_roles": [
            {
                "role": "avoid_reopening | acknowledge_fulfillment | keep_waiting",
                "anchor": "prompt-safe downstream semantic anchor",
            }
        ],
        "warnings": ["prompt-safe warning text"],
    }
}
```

This state key must never contain `unit_id`, collection names, repository
payloads, raw action specs, or database error text. Private executable action
specs and validation errors remain in action results and traces, not in L3
prompt payloads.

### Executable DB action

`apply_memory_lifecycle_update` is the only executable DB update shape:

```python
{
    "schema_version": "action_spec.v1",
    "kind": "apply_memory_lifecycle_update",
    "target": {
        "target_kind": "memory_unit",
        "target_id": "<trusted unit_id>",
        "owner": "user_memory_units",
        "scope": {"unit_type": "active_commitment"},
    },
    "params": {
        "memory_kind": "user_memory_unit",
        "unit_type": "active_commitment",
        "unit_id": "<trusted unit_id>",
        "lifecycle_decision": "fulfilled | abandoned | obsolete | deferred",
        "due_at": str | None,
    },
    "reason": "<specialist evidence_anchor>",
}
```

## LLM Call And Context Budget

Before this plan:

- Normal live chat uses L2d for action selection and selected L3/dialog only
  when `speak` is selected.
- L2d may directly materialize one executable `memory_lifecycle_update` only
  when exactly one active commitment exists.

After this plan:

- L2d still runs once per cognition episode.
- The memory lifecycle specialist adds one response-path LLM call only when
  L2d selects `memory_lifecycle_update`.
- The specialist prompt receives current input, four L2 fields, active
  commitment projections, bounded retrieved evidence, and conversation
  progress. The prompt must stay below 12k characters, well under the 50k token
  default context cap.
- No retry call, repair prompt, background loop, or extra summarizer is added.
- If active commitments are absent after L2d selects the route, deterministic
  code skips the specialist, records a private rejected/no-op result, and does
  not invent a lifecycle decision.

Latency impact is one additional LLM call only on selected memory lifecycle
turns. This is accepted because stale promise closure must happen before L3
content anchoring when the same turn also speaks.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`
  - Public entrypoint: `call_memory_lifecycle_update_handler(state:
    GlobalPersonaState) -> dict`.
  - Owns alias projection, specialist prompt/LLM call, output normalization,
    content-anchor role projection, and executable action-spec materialization.
- `tests/test_memory_lifecycle_specialist.py`
  - Deterministic and patched-LLM tests for alias binding, no-change
    normalization, forbidden ID leakage, and executable action construction.
- `tests/test_memory_lifecycle_specialist_live_llm.py`
  - One-case-at-a-time real LLM tests for tiramisu fulfilled, new sweet plan,
    magic fulfilled, and ambiguous reply.

### Modify

- `src/kazusa_ai_chatbot/action_spec/registry.py`
  - Keep `MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "memory_lifecycle_update"` as
    the route intent.
  - Add `APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY =
    "apply_memory_lifecycle_update"` as executable DB action.
  - Update prompt projections so L2d sees route semantics, not DB params.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
  - Validate `memory_lifecycle_update` as a specialist route intent.
  - Validate `apply_memory_lifecycle_update` with the existing DB lifecycle
    contract.
- `src/kazusa_ai_chatbot/action_spec/execution.py`
  - Execute only `apply_memory_lifecycle_update` through
    `execute_user_memory_lifecycle_action`.
  - Do not execute `memory_lifecycle_update` directly.
- `src/kazusa_ai_chatbot/action_spec/handlers/memory_lifecycle.py`
  - Rename validation expectations from `memory_lifecycle_update` to
    `apply_memory_lifecycle_update`.
  - Preserve repository update behavior and lifecycle enum mapping.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
  - Remove `_select_active_commitment(...)` as a prerequisite for exposing
    `memory_lifecycle_update`.
  - Remove `_build_memory_lifecycle_action_spec(...)` executable target
    binding.
  - Materialize route-intent specs with no DB target fields.
  - Update prompt wording so L2d selects review need, not lifecycle decision or
    target.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Insert memory lifecycle handler execution after cognition and before
    selected text L3/dialog.
  - Ensure private-only memory lifecycle episodes still execute and trace.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Pass prompt-safe memory lifecycle role anchors into selected L3 content
    anchor context.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add the prompt-safe `memory_lifecycle_context` state key defined in this
    plan.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Include prompt-safe memory lifecycle roles in the content-anchor agent
    payload and instructions.
- `src/kazusa_ai_chatbot/action_spec/README.md`,
  `src/kazusa_ai_chatbot/nodes/README.md`, and
  `src/kazusa_ai_chatbot/db/README.md`
  - Update ownership docs and action-kind names.
- Existing tests that assert old executable `memory_lifecycle_update` behavior:
  `tests/test_action_spec_memory_lifecycle.py`,
  `tests/test_action_spec_evaluator.py`,
  `tests/test_action_spec_results.py`,
  `tests/test_persona_supervisor2_action_initializer.py`,
  `tests/test_l2d_action_selection_cases.py`,
  `tests/test_l2d_action_selection_live_llm.py`,
  `tests/test_self_cognition_memory_lifecycle_live_llm.py`,
  `tests/test_self_cognition_tracking.py`,
  `tests/test_consolidator_facts_rag2.py`,
  `tests/test_service_background_consolidation.py`.

### Keep

- `src/kazusa_ai_chatbot/db/user_memory_units.py`
  - Keep existing lifecycle update repository semantics.
- Existing `LIFECYCLE_STATUS_BY_DECISION`
  - Keep enum mapping from LLM lifecycle decisions to DB statuses.
- Public adapter and `/chat` response contracts.

## Implementation Order

1. Add focused action-spec tests for the kind split.
   - File: `tests/test_action_spec_memory_lifecycle.py`.
   - Expected before implementation: tests fail because
     `apply_memory_lifecycle_update` is unsupported.
2. Update action-spec registry, evaluator, execution, and handler rename.
   - Verify focused action-spec tests pass.
3. Add patched specialist tests.
   - File: `tests/test_memory_lifecycle_specialist.py`.
   - Expected before implementation: tests fail because the module does not
     exist.
4. Implement `persona_supervisor2_memory_lifecycle.py`.
   - Verify patched specialist tests pass.
5. Update L2d prompt and materialization.
   - Update `tests/test_persona_supervisor2_action_initializer.py`.
   - Verify L2d route specs contain no `unit_id`.
6. Integrate the specialist into `persona_supervisor2.py`.
   - Add or update integration tests that prove memory routing runs before L3
     text and private-only episodes execute.
7. Update L3 content-anchor payload consumption.
   - Verify content-anchor tests include memory lifecycle role anchors and no
     raw IDs.
8. Update self-cognition and consolidation tests to the new executable action
   kind.
9. Add real LLM specialist tests and run them one at a time.
10. Update docs and run static greps.
11. Run the full Verification section.
12. Run Independent Code Review and remediate findings.

## Progress Checklist

- [ ] Stage 1 - action-spec kind split complete
  - Covers: Implementation Order steps 1-2.
  - Verify: focused action-spec tests pass.
  - Evidence: record failing baseline, changed files, and passing output in
    Execution Evidence.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - specialist module contract complete
  - Covers: Implementation Order steps 3-4.
  - Verify: `tests/test_memory_lifecycle_specialist.py` passes.
  - Evidence: record patched LLM test output and prompt-safety assertions.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - L2d router contract complete
  - Covers: Implementation Order step 5.
  - Verify: L2d focused tests pass and prompt grep shows no DB target binding
    language.
  - Evidence: record test output and grep result.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - persona graph and L3 handoff complete
  - Covers: Implementation Order steps 6-7.
  - Verify: integration tests pass and memory lifecycle role anchors reach
    content-anchor payload without `unit_id`.
  - Evidence: record test output and inspected payload.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - self-cognition, consolidation, docs, and live tests complete
  - Covers: Implementation Order steps 8-10.
  - Verify: listed focused tests pass; real LLM tests are run one at a time and
    traces are inspected.
  - Evidence: record command output, trace paths, and manual judgment.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - full verification and independent code review complete
  - Covers: Implementation Order steps 11-12.
  - Verify: all Verification commands pass and Independent Code Review is
    approved.
  - Evidence: record full command output, review findings, fixes, reruns, and
    residual risks.
  - Sign-off: `<agent/date>` after review approval and evidence are recorded.

## Verification

### Static Greps

- `rg "\"memory_lifecycle_update\"" src tests`
  - Expected: matches remain only for router intent, docs, tests, and
    non-executable capability definitions.
  - Forbidden: executable DB handler validation, repository execution, or
    `unit_id` params under kind `memory_lifecycle_update`.
- `rg "\"apply_memory_lifecycle_update\"" src tests`
  - Expected: matches in action-spec registry, evaluator, execution, memory
    lifecycle handler, docs, and executable-action tests.
- `rg "unit_id|target_alias" src\\kazusa_ai_chatbot\\nodes\\persona_supervisor2_cognition_l2d.py`
  - Expected: no prompt-facing L2d lifecycle target binding. Matches are
    allowed only if unrelated source-ref helpers remain for other actions.
- `rg "memory_lifecycle_specialist|memory_lifecycle_context" src tests`
  - Expected: specialist module, persona graph integration, L3 payload
    consumption, tests, and docs.

### Focused Deterministic And Patched Tests

- `venv\\Scripts\\python.exe -m pytest tests\\test_action_spec_memory_lifecycle.py -q`
- `venv\\Scripts\\python.exe -m pytest tests\\test_action_spec_evaluator.py -q`
- `venv\\Scripts\\python.exe -m pytest tests\\test_action_spec_results.py -q`
- `venv\\Scripts\\python.exe -m pytest tests\\test_memory_lifecycle_specialist.py -q`
- `venv\\Scripts\\python.exe -m pytest tests\\test_persona_supervisor2_action_initializer.py -q`
- `venv\\Scripts\\python.exe -m pytest tests\\test_self_cognition_tracking.py -q`
- `venv\\Scripts\\python.exe -m pytest tests\\test_service_background_consolidation.py -q`
- Include patched specialist tests for 13 active commitments, proving only 12
  prompt-safe aliases are exposed and omitted rows remain unchanged.
- Include patched specialist tests for more than 3 valid lifecycle decisions,
  proving only the first 3 priority-ordered updates are materialized and
  surplus decisions are logged.

### Integration Tests

- `venv\\Scripts\\python.exe -m pytest tests\\test_cognition_stage_connection_live_llm.py::test_live_cognition_stage_connection_case -q -s`
  - Run only with an explicit case file and case id when doing live stage
    connection verification.
- Add and run a patched persona graph integration test proving:
  - L2d selects `memory_lifecycle_update`.
  - The specialist runs before selected L3 text.
  - The handler consumes the router intent and emits
    `apply_memory_lifecycle_update`.
  - Content-anchor payload receives prompt-safe role anchors.

### Real LLM Tests

Run each live case one at a time and inspect the trace before running the next:

- `venv\\Scripts\\python.exe -m pytest tests\\test_memory_lifecycle_specialist_live_llm.py::test_live_tiramisu_fulfilled -q -s`
- `venv\\Scripts\\python.exe -m pytest tests\\test_memory_lifecycle_specialist_live_llm.py::test_live_new_sweet_plan_no_lifecycle_change -q -s`
- `venv\\Scripts\\python.exe -m pytest tests\\test_memory_lifecycle_specialist_live_llm.py::test_live_magic_fulfilled -q -s`
- `venv\\Scripts\\python.exe -m pytest tests\\test_memory_lifecycle_specialist_live_llm.py::test_live_ambiguous_reply_no_lifecycle_change -q -s`
- `venv\\Scripts\\python.exe -m pytest tests\\test_memory_lifecycle_specialist_live_llm.py::test_live_capacity_12_commitments_last_alias_fulfilled -q -s`

Expected manual judgment: the specialist selects the correct alias and
lifecycle decision for fulfilled cases, emits no lifecycle update for no-change
cases, and never leaks `unit_id` in raw model output.

### Syntax And Style

- `venv\\Scripts\\python.exe -m py_compile src\\kazusa_ai_chatbot\\nodes\\persona_supervisor2_memory_lifecycle.py`
- Run py_compile on every edited Python file containing CJK prompt text.

## Independent Plan Review

Gate run on 2026-05-18 before approval. No separate reviewer was available in
this session, so the active agent reread the plan-writing contract, execution
gates, cutover policy, registry, current plan, and relevant experiment evidence
from a fresh-review posture.

Review inputs:

- `development_plans/README.md`
- `.agents/skills/development-plan-writing/references/plan_contract.md`
- `.agents/skills/development-plan-writing/references/execution_gates.md`
- `.agents/skills/development-plan-writing/references/cutover_policy.md`
- `experiments/l2d_commitment_alias_experiment.py`
- `test_artifacts/llm_traces/l2d_commitment_alias_experiment__20260517T140754819496Z.json`
- `test_artifacts/llm_traces/l2d_commitment_alias_experiment__20260517T200937403540Z.json`

Review rule:

```text
L2d routes memory_lifecycle_update.
The specialist decides target_alias and lifecycle decision.
Deterministic code resolves alias and applies DB update.
```

Findings and fixes:

| Severity | Finding | Fix |
|---|---|---|
| Blocker | L2d router action included `decision_hint`, which allowed L2d to shape lifecycle content beyond route selection. | Removed `decision_hint`; L2d may emit only `review_kind` and semantic `detail`. |
| Blocker | `memory_lifecycle_context` was named in Change Surface but lacked an exact prompt-safe state shape. | Added the state contract and explicitly banned `unit_id`, repository payloads, raw action specs, and DB error text. |
| Blocker | The production verification list did not include the live 12-alias capacity case proven by experiment. | Added `test_live_capacity_12_commitments_last_alias_fulfilled` to the real LLM verification gate. |
| Non-blocking | The capacity proof currently lives under ignored `experiments/` and `test_artifacts/`; implementation still needs tracked pytest coverage. | Kept the evidence in Context and Verification; the approved plan requires tracked live and patched tests. |

Self-review result: coverage, minimality, placeholder scan, contract
consistency, checklist granularity, and verification mapping pass after the
blocker fixes above.

Approval status: approved for implementation after the fixes above. Remaining
risk is implementation risk, not plan-readiness risk.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including hidden fallback paths, old
  executable `memory_lifecycle_update` residue, prompt/RAG payload leaks,
  persistence risk, fixture brittleness, and avoidable blast radius.
- Alignment with Must Do, Deferred, Agent Autonomy Boundaries, Change Surface,
  exact contracts, implementation order, verification gates, and acceptance
  criteria.
- Regression and handoff quality, including real LLM traces, focused tests,
  static greps, execution evidence, and next-stage handoff notes.

Fix findings directly only when the fix is inside this plan's change surface.
If a finding requires a new contract, fallback path, or change outside the
approved boundary, stop and update the plan or request approval before editing
code.

## Acceptance Criteria

This plan is complete when:

- L2d `memory_lifecycle_update` action specs contain no DB target, `unit_id`,
  lifecycle target alias, or repository params.
- The old executable DB action kind has been renamed to
  `apply_memory_lifecycle_update` across source, tests, and docs.
- A dedicated memory lifecycle specialist can bind one of multiple active
  commitments by alias and produce an executable update through deterministic
  alias resolution.
- Specialist capacity is explicit and tested: up to 12 reviewed commitments per
  call and up to 3 materialized lifecycle updates per user turn.
- The original tiramisu fulfillment case produces
  `commitment_1 -> fulfilled -> apply_memory_lifecycle_update` and a
  prompt-safe role anchor that prevents reopening “提拉米苏还没兑现”.
- No-change cases do not produce executable lifecycle updates.
- Selected `speak` behavior still goes through L3/dialog, and L3 content
  anchors can consume memory lifecycle role anchors.
- Private-only memory lifecycle updates still execute and are represented in
  action results and episode trace.
- All Verification gates pass and Independent Code Review is approved.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Added response-path latency | Specialist runs only when L2d selects the route; no retries | Live LLM trace timings and manual inspection |
| L2d still emits executable DB fields | Remove old materializer and add prompt/action-spec tests | L2d focused tests and static greps |
| Specialist leaks raw IDs | Prompt receives aliases only; validation rejects unknown fields where relevant | Patched tests and live trace inspection |
| False-positive lifecycle closure | Specialist no-change cases and no keyword fallback | Live and patched no-change tests |
| Self-cognition lifecycle closure breaks | Update self-cognition tests to use new executable kind after owned target exists | `tests/test_self_cognition_tracking.py` |
| Action execution ignores new kind | Rename execution branch and tests | `tests/test_action_spec_results.py` |

## Data Migration

No database migration is required. Existing `user_memory_units` documents,
lifecycle statuses, merge history, and active-commitment rows remain unchanged.

## Execution Evidence

- Pre-implementation experiment evidence:
  - `venv\Scripts\python.exe -m py_compile experiments\l2d_commitment_alias_experiment.py`
    passed.
  - `venv\Scripts\python.exe experiments\l2d_commitment_alias_experiment.py capacity_12_commitments_last_alias_fulfilled`
    passed.
  - Trace:
    `test_artifacts/llm_traces/l2d_commitment_alias_experiment__20260517T200937403540Z.json`.
- Independent Plan Review completed on 2026-05-18. Three blockers were found
  and fixed in this plan before approval.
- Implementation not started. All Progress Checklist items remain unchecked.

## Execution Handoff

Execution mode after approval: inline execution in the current workspace by one
implementation agent. The implementation agent starts at Stage 1, loads the
Mandatory Skills, rereads this entire plan, and records the initial failing
action-spec test evidence before editing production code.
