# cognition llm stage reconnection plan

## Summary

- Goal: reconnect all cognition LLM stages after the L2 shuffle so social
  context is available to L2d action selection, selected L3/L4 text surface
  handling still works, and the full cognition module is validated with real
  QQ private and group conversation cases for user `673225019`.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `database-data-pull`, `cjk-safety`
- Overall cutover strategy: bigbang internal graph and naming cutover; stable
  public `/chat` response shape
- Highest-risk areas: cognition graph joins, prompt-selection stage names,
  L2d input payload shape, selected L3 text-surface state wiring, L4 collector
  input completeness, live LLM test artifacts from real QQ private/group
  history, and accidental LLM responsibility drift
- Acceptance criteria: contextual/social appraisal runs as `L2c2` before L2d;
  L2d receives social context without gaining new responsibility; selected L3
  text and L4 collector consume the existing social fields without rerunning
  contextual; deterministic handoff tests pass; real LLM integration traces
  exist for QQ user `673225019` private and group cases; all LLMs are connected
  through one inspectable cognition path

This plan is a follow-up architecture correction discovered during execution
of `modality_neutral_action_spec_effector_expansion_plan.md` and
`l2d_l3_surface_handoff_plan.md`.

## Context

The accepted architecture is:

```text
typed episode
  -> RAG/evidence
  -> L1 affect_appraisal
  -> L2a conscious_framing + L2b boundary_appraisal
  -> L2c1 judgment_synthesis + L2c2 social_context_appraisal
  -> L2d action_selection
  -> selected L3 surfaces / action handlers
  -> L4 surface directive collector
  -> dialog / action results / surface outputs
  -> episode-trace consolidation
```

The key design decision is fixed:

```text
No LLM responsibility changes.
Only graph placement, state handoff, and internal naming change.
```

The contextual LLM keeps the same semantic job it already has: produce
`social_distance`, `emotional_intensity`, `vibe_check`, and
`relational_dynamic`. It does not decide whether to reply, does not choose
actions, does not emit `expression_willingness`, does not emit action specs,
and does not generate final wording. The only change is that this existing
social-context appraisal runs earlier as L2 residue so L2d can use it when
selecting zero, one, or many actions.

The current code has already moved selected L3 text handling behind L2d in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`, but that
handler still calls `call_contextual_agent(...)` as an L3 node. After this
plan, L3 text receives social context from cognition state and runs only the
surface-specific LLMs plus the deterministic collector.

## Mandatory Skills

- `development-plan-writing`: load before changing this plan, registry rows,
  execution evidence, lifecycle status, or plan sign-off.
- `local-llm-architecture`: load before changing cognition graph routing,
  prompt-selection stage names, L2d payload shape, L3/L4 handoff, dialog
  integration, or live LLM test design.
- `no-prepost-user-input`: load before touching commitment, preference,
  permission, promise, or action-selection behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before exporting or handling real QQ conversation
  history from MongoDB.
- `cjk-safety`: load before editing Python files containing CJK prompt strings.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.
- Use `venv\Scripts\python` or `venv\Scripts\python.exe` for Python commands.
- Do not read `.env` directly. Let the project scripts load environment
  settings.
- Do not change any LLM responsibility under this plan.
- Do not add new LLM output fields to represent action authority, response
  willingness, expression posture, delivery permission, adapter feasibility,
  database targets, or scheduler targets.
- Do not reintroduce `expression_willingness`, `expression_posture`,
  `silent`, `no_response`, or equivalent response-decision fields in L3.
- Do not make contextual/social appraisal choose actions. It only emits the
  existing four social context fields.
- Do not make L2a, L2b, L2c1, L2c2, L3, L4, dialog, RAG, or consolidation emit
  `ActionSpecV1`. L2d remains the only LLM action selector.
- Do not make L3 text or dialog decide whether to speak. If L3/dialog is
  called, `speak` was already selected by L2d.
- Do not make a tool/action handler call cognition inline.
- Do not expose raw platform IDs, handler IDs, schema versions, source-ref IDs,
  database collection names, scheduler IDs, or persistence IDs to LLM prompts.
- Preserve the existing prompt responsibilities. Prompt edits are limited to
  stage-name references, prompt-selection keys, and the minimal payload labels
  required by graph relocation.
- Preserve existing public service response shape. This plan changes internal
  graph wiring and naming only.
- Real LLM tests must run one case at a time and each emitted trace must be
  inspected before running the next case.
- Real QQ history artifacts belong under `test_artifacts/` and must not be
  committed. Do not quote large message contents in plan evidence or final
  chat responses.

## Must Do

- Rename internal cognition stage labels:
  - `l2a_consciousness` -> `l2a_conscious_framing`
  - `l2b_boundary_core` -> `l2b_boundary_appraisal`
  - `l2c_judgment_core` -> `l2c1_judgment_synthesis`
  - `l3_contextual_agent` -> `l2c2_social_context_appraisal`
  - `l2d_action_initializer` -> `l2d_action_selection`
  - `l4_collector` -> `l4_surface_directive_collector`
- Move the existing contextual LLM call from selected L3 text handling into the
  cognition subgraph as `L2c2 social_context_appraisal`.
- Keep the contextual prompt responsibility unchanged and keep its four output
  fields unchanged.
- Wire the cognition subgraph as:

```text
L1 -> L2a and L2b
L2a + L2b -> L2c1
L2b -> L2c2
L2c1 + L2c2 -> L2d
L2d -> END
```

- Add the existing social context fields to the L2d prompt payload as
  prompt-safe L2 residue. L2d remains action selection only.
- Update selected L3 text handling so it no longer calls contextual/social
  appraisal. It must receive `social_distance`, `emotional_intensity`,
  `vibe_check`, and `relational_dynamic` from the post-L2d global state.
- Ensure L3 style, content-anchor, preference, visual side-effect, and L4
  collector still run correctly when `speak` is selected.
- Ensure no selected `speak` skips L3, L4, and dialog while still producing
  private trace/consolidation evidence.
- Update prompt-selection stage names, output-contract stage names, schema
  annotations, tests, docs, and static greps to match the new naming.
- Add deterministic graph and handoff tests for:
  - L2c2 runs before L2d;
  - L2d sees L2c2 output;
  - L3 text surface does not call L2c2;
  - L3 visual and L4 collector receive L2c2 fields;
  - selected `speak` runs L3/L4/dialog once;
  - no selected `speak` skips L3/L4/dialog and still emits private trace.
- Add real LLM integration fixtures and tests for the full cognition module
  using real QQ conversation history for platform user `673225019`:
  - private chat case set;
  - group chat case set;
  - historical assistant reply comparison;
  - per-stage LLM trace output;
  - final route/surface/dialog trace output.
- Record real LLM evidence case by case. Do not batch live LLM tests for a
  green/red summary.

## Deferred

- Do not change the semantic responsibility of any existing LLM prompt.
- Do not add standalone L3 image action routing.
- Do not call an external image-generation service.
- Do not implement new tools such as web research, note-taking, arbitrary
  messaging, file writes, shell calls, or MongoDB writes.
- Do not rename public service response fields.
- Do not migrate MongoDB data.
- Do not change platform adapters.
- Do not broaden self-cognition scope beyond regression coverage needed by
  existing action-spec plans.
- Do not implement prompt retry or repair loops.
- Do not preserve old internal stage names through compatibility aliases.

## Cutover Policy

Overall strategy: bigbang internal graph and naming cutover; stable public
service response shape.

| Area | Policy | Instruction |
|---|---|---|
| Internal cognition stage names | bigbang | Rename internal node labels, prompt-selection keys, contracts, docs, and tests. Do not keep old aliases in runtime code. |
| Contextual/social appraisal placement | bigbang | Move the existing contextual LLM call to `L2c2`. Remove it from selected L3 text. |
| LLM responsibilities | stable | No LLM semantic responsibility changes. Prompt edits must only support placement and naming. |
| L2d input payload | compatible internal | Add prompt-safe social context residue. Do not add execution params or handler details. |
| Selected L3 text handler | bigbang | Consume social context from state. Do not rerun contextual. |
| L4 collector | bigbang internal | Rename and wire collector after selected L3 surface outputs. Preserve collected `action_directives` shape. |
| Public `/chat` response | compatible | Preserve current public response fields and behavior shape. |
| Real LLM test artifacts | additive | Store under `test_artifacts/cognition_stage_connection/`; do not commit artifacts. |

## Cutover Policy Enforcement

- If an area is marked `bigbang`, delete or rewrite old internal references.
  Do not add aliases, shims, fallback branches, or dual graph paths.
- If a stage name appears in prompt-selection contracts, output contracts,
  docs, tests, and graph node labels, all copies must be updated in the same
  stage.
- If implementation discovers a required compatibility surface outside public
  `/chat` shape, stop and update this plan before adding it.

## Agent Autonomy Boundaries

- The implementation agent may choose local helper names only when the helper
  preserves the graph and ownership decisions in this plan.
- The implementation agent must not invent new LLM outputs, new action
  capabilities, new prompt responsibilities, new graph branches, new feature
  flags, or new fallback paths.
- Changes outside `src/kazusa_ai_chatbot/nodes/`,
  `src/kazusa_ai_chatbot/cognition_episode.py`,
  `src/scripts/`, `tests/`, and relevant `development_plans/` docs require a
  plan update before implementation.
- Existing dirty worktree changes must be preserved unless they conflict with
  this plan. Do not revert user or prior-agent changes without explicit user
  instruction.
- If existing tests use old stage names only as historical labels, update the
  labels. Do not keep runtime aliases to satisfy tests.

## Target State

After execution, the current live cognition path is:

```text
stage_0_msg_decontexualizer
  -> stage_1_research
  -> stage_2_cognition
       L1 affect_appraisal
       L2a conscious_framing
       L2b boundary_appraisal
       L2c1 judgment_synthesis
       L2c2 social_context_appraisal
       L2d action_selection
  -> action router
       speak selected
         -> selected L3 text surface
              interaction-style context loader
              style agent
              content-anchor agent
              preference adapter
              visual side-effect agent
              L4 surface directive collector
         -> dialog
         -> text SurfaceOutputV1 + speak ActionResultV1
       no speak selected
         -> no L3/L4/dialog
         -> private SurfaceOutputV1/action results when applicable
  -> episode trace
  -> consolidation
```

The social context fields remain:

```python
{
    "social_distance": str,
    "emotional_intensity": str,
    "vibe_check": str,
    "relational_dynamic": str,
}
```

L2d receives those fields as social context evidence. It still emits only
semantic `action_requests`, which deterministic code materializes into action
specs.

L3/L4 receives those fields from state. It does not call the social-context LLM
again.

## Design Decisions

1. Contextual/social appraisal moves because L2d needs social-temperature
   evidence before action selection.
2. Contextual/social appraisal does not gain any new authority. It remains the
   same LLM job with the same four output fields.
3. L2c2 runs after L2b because the current contextual prompt already consumes
   `boundary_core_assessment`.
4. L2c1 and L2c2 join before L2d. This keeps judgment and social context as
   sibling L2 residue, not surface rendering.
5. L3 text remains selected-surface work. It owns style, content anchors,
   preference adaptation, visual side-effect directives, and the collector
   input for dialog.
6. L4 remains deterministic collection, not an LLM responsibility.
7. Real LLM validation uses frozen real conversation seeds after
   decontextualization/RAG context assembly, then runs the live cognition
   module so LLM cooperation is tested without making RAG retrieval variance
   part of this plan's pass/fail surface.

## Contracts And Data Shapes

### Cognition State Additions

The cognition state already carries the social fields. This plan makes their
producer `L2c2` instead of selected L3 text.

```python
social_distance: str
emotional_intensity: str
vibe_check: str
relational_dynamic: str
```

### L2d Payload Addition

`build_action_initializer_payload(...)` must include:

```python
"social_context_appraisal": {
    "social_distance": state["social_distance"],
    "emotional_intensity": state["emotional_intensity"],
    "vibe_check": state["vibe_check"],
    "relational_dynamic": state["relational_dynamic"],
}
```

This field is prompt-safe semantic residue. It is not permission, routing,
delivery, target binding, adapter feasibility, or persistence data.

### Selected L3 Text Input

`call_l3_text_surface_handler(...)` must seed the selected L3 graph with:

```python
"social_distance": state["social_distance"],
"emotional_intensity": state["emotional_intensity"],
"vibe_check": state["vibe_check"],
"relational_dynamic": state["relational_dynamic"],
```

The selected L3 graph must not include a contextual/social LLM node.

### Real LLM Case Set

Create a private, uncommitted case-set format under
`test_artifacts/cognition_stage_connection/`.

```python
{
    "schema_version": "cognition_stage_connection_case_set.v1",
    "source_platform": "qq",
    "source_platform_user_id": "673225019",
    "created_at": "ISO-8601 UTC",
    "cases": [
        {
            "schema_version": "cognition_stage_connection_case.v1",
            "case_id": "qq_private_001",
            "source_kind": "qq_private | qq_group",
            "source_channel_type": "private | group",
            "source_channel_id": "prompt-safe channel label or omitted",
            "seed_state": {},
            "historical_user_message": "real message text",
            "historical_assistant_reply": ["real assistant reply fragments"],
            "historical_comparison": {
                "expected_visible_surface": true,
                "comparison_basis": "assistant replied in stored history"
            }
        }
    ]
}
```

The case artifact may include real message text because it is local
test-artifact data, but it must not include embeddings, raw credentials, or
unbounded chat history.

## LLM Call And Context Budget

- The L2 shuffle must not add a new LLM call. It moves the existing contextual
  call from selected L3 text to L2c2.
- For visible text cases, the total live call count remains equivalent to the
  previous selected-text path: L1, L2a, L2b, L2c1, L2c2, L2d, selected L3 text
  agents, and dialog.
- For no-speak cases, the call count improves because selected L3 and dialog
  do not run.
- Do not add prompt retries, repair calls, or second-pass validators for this
  plan.
- Live LLM test cases run one at a time. The full QQ private/group evidence
  pass is intentionally diagnostic and must write durable trace files.

## Change Surface

### Source

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Rename graph nodes.
  - Add `L2c2` social-context node after `L2b`.
  - Join `L2c1` and `L2c2` into `L2d`.
  - Return social context fields with L2/L2d state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Rename `call_contextual_agent(...)` to the L2c2 social-context function.
  - Preserve prompt responsibility and output fields.
  - Rename `call_collector(...)` to the L4 surface directive collector.
  - Update validation stage names.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Remove contextual/social node from selected L3 graph.
  - Seed social context from state.
  - Rename collector node.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Rename stage labels for L2a/L2b/L2c1 validation and prompt selection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
  - Rename stage label to `l2d_action_selection`.
  - Add `social_context_appraisal` to prompt payload.
  - Keep semantic output contract unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  - Replace stage names and prompt keys.
  - Treat `l2c2_social_context_appraisal` as using the same variants and prompt
    body formerly used by `l3_contextual_agent`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
  - Rename output-contract stage keys.
  - Keep contextual/social output fields unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Update comments/type names where they refer to old stage ownership.
- `src/scripts/capture_cognition_stage_connection_cases.py`
  - New script for local real-history case capture.

### Tests

- `tests/test_cognition_stage_connection.py`
  - New deterministic graph and state-handoff tests.
- `tests/test_cognition_stage_connection_live_llm.py`
  - New one-case real LLM full cognition module test.
- `tests/cognition_stage_connection_cases.py`
  - New loader/comparison helpers for local case sets.
- Existing tests that import old stage names must be updated to new names.
- Existing L2d and L3 handoff tests must be updated to assert the new L2c2
  placement and the selected-L3 state contract.

### Docs And Plans

- `src/kazusa_ai_chatbot/nodes/README.md`
  - Update graph diagram, L2/L3/L4 role table, and stage names.
- `development_plans/reference/designs/cognition_contracts_design.md`
  - Add the L2c1/L2c2 naming and state that this is a placement/naming change,
    not an LLM responsibility change.
- `development_plans/active/short_term/l2d_l3_surface_handoff_plan.md`
  - Add a reconciliation note that contextual/social appraisal is now L2c2.
- `development_plans/active/short_term/modality_neutral_action_spec_effector_expansion_plan.md`
  - Add an execution-note reference to this plan as the LLM connection cleanup.

## Data Source And Artifacts

Use read-only exports first. Do not query MongoDB ad hoc from shell.

Private history export:

```powershell
venv\Scripts\python.exe -m scripts.export_collection conversation_history --filter '{"platform":"qq","platform_user_id":"673225019","channel_type":"private"}' --sort '{"timestamp":-1}' --limit 500 --output test_artifacts\cognition_stage_connection\qq_673225019_private_history.json
```

Group history export:

```powershell
venv\Scripts\python.exe -m scripts.export_collection conversation_history --filter '{"platform":"qq","platform_user_id":"673225019","channel_type":"group"}' --sort '{"timestamp":-1}' --limit 500 --output test_artifacts\cognition_stage_connection\qq_673225019_group_history.json
```

Case capture:

```powershell
venv\Scripts\python.exe src\scripts\capture_cognition_stage_connection_cases.py --private-export test_artifacts\cognition_stage_connection\qq_673225019_private_history.json --group-export test_artifacts\cognition_stage_connection\qq_673225019_group_history.json --max-private 20 --max-group 20 --output-dir test_artifacts\cognition_stage_connection
```

The capture script must select user-message rows by QQ user `673225019` that
have a nearby historical assistant reply in the same conversation. The
historical assistant reply is the comparison baseline for visible-surface
routing and qualitative content alignment. If either the private or group
export contains no valid user/assistant pairs, execution stops and records a
data blocker.

## Overdesign Guardrail

- Actual problem: the contextual/social LLM is connected as selected L3 text,
  but L2d needs its social-context evidence before choosing actions.
- Minimal change: move the existing contextual call earlier as `L2c2`, pass its
  same four fields into L2d and selected L3, rename internal stages, and verify
  full graph connectivity.
- Ownership boundaries: L2c2 owns social context appraisal; L2d owns semantic
  action selection; selected L3 owns surface directives; L4 owns deterministic
  collection; dialog owns final text; deterministic code owns routing,
  validation, traces, scheduling, delivery, persistence, and audit.
- Rejected complexity: no new LLM responsibilities, no new action kinds, no
  schema-driven prompt expansion, no compatibility aliases, no generic chaining
  engine, no prompt repair loop, no public API change, no database migration.
- Evidence threshold for future expansion: standalone image surfaces, web
  research, note actions, external sends, or generic action chains require a
  separate approved plan with capability ownership, permissions, trace shape,
  and real LLM/router tests.

## Implementation Order

1. Add deterministic failing tests for L2c2/L2d/L3/L4 graph connection.
2. Update internal stage names and prompt-selection/output-contract names.
3. Move contextual/social appraisal into cognition as L2c2.
4. Add social context projection to the L2d payload.
5. Rewire selected L3 text to consume social context from state and remove its
   contextual node.
6. Update L4 collector naming and selected L3/L4 tests.
7. Update docs and active-plan reconciliation notes.
8. Add real QQ history capture script and case-set helpers.
9. Add one-case live LLM integration test for full cognition connection.
10. Export private and group QQ history for user `673225019`, capture case
    sets, and run live LLM cases one at a time.
11. Run static greps, deterministic tests, focused live tests, and independent
    code review.

## Progress Checklist

- [ ] Stage 1 - deterministic graph contract tests added
  - Covers: tests proving L2c2 runs before L2d, L2d sees social context, L3 no
    longer calls contextual, and L4 still collects social fields.
  - Files: `tests/test_cognition_stage_connection.py`,
    `tests/test_l2d_l3_surface_handoff.py`.
  - Verify before implementation:
    `venv\Scripts\python -m pytest tests\test_cognition_stage_connection.py tests\test_l2d_l3_surface_handoff.py -q`.
  - Expected evidence before implementation: at least one failure showing the
    current graph still calls contextual from selected L3 or does not feed L2d
    social context.
  - Sign-off: record failure output in `Execution Evidence`.

- [ ] Stage 2 - stage names and contracts updated
  - Covers: prompt-selection stage literals, output contract keys, schema
    comments, graph node labels, and test references.
  - Files: `persona_supervisor2_cognition_prompt_selection.py`,
    `persona_supervisor2_cognition_output_contracts.py`,
    `persona_supervisor2_schema.py`, affected tests.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_persona_supervisor2_schema.py -q`.
  - Sign-off: record changed names and test output.

- [ ] Stage 3 - L2c2 social context appraisal wired before L2d
  - Covers: cognition graph edge rewrite and L2d payload update.
  - Files: `persona_supervisor2_cognition.py`,
    `persona_supervisor2_cognition_l3.py`,
    `persona_supervisor2_cognition_l2d.py`.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_cognition_stage_connection.py tests\test_persona_supervisor2_action_initializer.py -q`.
  - Sign-off: record graph edge summary and test output.

- [ ] Stage 4 - selected L3/L4 reconnected after L2 shuffle
  - Covers: selected L3 text no longer calls contextual; L3/L4 consume social
    context from state; dialog receives complete `action_directives`.
  - Files: `persona_supervisor2_l3_surface.py`,
    `persona_supervisor2_cognition_l3.py`, dialog-facing tests.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_l2d_l3_surface_handoff.py tests\test_dialog_agent.py tests\test_persona_supervisor2.py -q`.
  - Sign-off: record selected-surface and no-speak test output.

- [ ] Stage 5 - documentation and active-plan reconciliation complete
  - Covers: node README, contracts reference, and active plan notes.
  - Files: `src/kazusa_ai_chatbot/nodes/README.md`,
    `development_plans/reference/designs/cognition_contracts_design.md`,
    `development_plans/active/short_term/l2d_l3_surface_handoff_plan.md`,
    `development_plans/active/short_term/modality_neutral_action_spec_effector_expansion_plan.md`.
  - Verify:
    `rg -n "l3_contextual_agent|l2c_judgment_core|l2d_action_initializer|l4_collector" src\kazusa_ai_chatbot\nodes tests -g "*.py" -g "*.md"`.
  - Expected evidence: only intentional historical references in active plans
    or execution evidence remain.
  - Sign-off: record grep output and doc summary.

- [ ] Stage 6 - real QQ case capture implemented
  - Covers: read-only export workflow, case capture script, case loader, and
    trace helper integration.
  - Files: `src/scripts/capture_cognition_stage_connection_cases.py`,
    `tests/cognition_stage_connection_cases.py`.
  - Verify:
    `venv\Scripts\python src\scripts\capture_cognition_stage_connection_cases.py --help`;
    `venv\Scripts\python -m py_compile src\scripts\capture_cognition_stage_connection_cases.py tests\cognition_stage_connection_cases.py`.
  - Sign-off: record command output.

- [ ] Stage 7 - real LLM integration test added
  - Covers: one-case live LLM full cognition module test with per-stage trace.
  - Files: `tests/test_cognition_stage_connection_live_llm.py`.
  - Verify:
    `venv\Scripts\python -m py_compile tests\test_cognition_stage_connection_live_llm.py`.
  - Sign-off: record command output.

- [ ] Stage 8 - QQ private/group real LLM evidence captured and inspected
  - Covers: export, capture, and one-by-one live LLM runs for private and group
    cases.
  - Commands:
    `venv\Scripts\python.exe -m scripts.export_collection conversation_history --filter '{"platform":"qq","platform_user_id":"673225019","channel_type":"private"}' --sort '{"timestamp":-1}' --limit 500 --output test_artifacts\cognition_stage_connection\qq_673225019_private_history.json`;
    `venv\Scripts\python.exe -m scripts.export_collection conversation_history --filter '{"platform":"qq","platform_user_id":"673225019","channel_type":"group"}' --sort '{"timestamp":-1}' --limit 500 --output test_artifacts\cognition_stage_connection\qq_673225019_group_history.json`;
    `venv\Scripts\python.exe src\scripts\capture_cognition_stage_connection_cases.py --private-export test_artifacts\cognition_stage_connection\qq_673225019_private_history.json --group-export test_artifacts\cognition_stage_connection\qq_673225019_group_history.json --max-private 20 --max-group 20 --output-dir test_artifacts\cognition_stage_connection`.
  - Live verify shape for each case:
    `COGNITION_CONNECTION_CASE_FILE=<case-file> COGNITION_CONNECTION_CASE_ID=<case-id> venv\Scripts\python -m pytest tests\test_cognition_stage_connection_live_llm.py::test_live_cognition_stage_connection_case -q -s -m live_llm`.
  - Evidence: record each trace path, case id, source kind, selected action
    kinds, whether L3/L4/dialog ran, and manual judgment.
  - Sign-off: record private and group case counts plus trace paths.

- [ ] Stage 9 - final regression and independent code review complete
  - Covers: deterministic regression, static stale-reference scans, live LLM
    evidence review, and independent code review.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_cognition_stage_connection.py tests\test_l2d_l3_surface_handoff.py tests\test_persona_supervisor2.py tests\test_dialog_agent.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_persona_supervisor2_schema.py -q`;
    `rg -n "expression_willingness|expression_posture|conditional_skip_dialog_agent|_cognition_requests_silence" src\kazusa_ai_chatbot\nodes tests -g "*.py" -g "*.md"`;
    `rg -n "l2d_action_initializer.*l3_|l3_.*l2d_action_initializer" src\kazusa_ai_chatbot\nodes tests -g "*.py"`.
  - Sign-off: record review result, remediations, rerun commands, and residual
    risks.

## Verification

### Deterministic Verification

```powershell
venv\Scripts\python -m pytest tests\test_cognition_stage_connection.py tests\test_l2d_l3_surface_handoff.py -q
venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_dialog_agent.py -q
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_persona_supervisor2_schema.py -q
```

### Static Contract Checks

```powershell
rg -n "expression_willingness|expression_posture|conditional_skip_dialog_agent|_cognition_requests_silence" src\kazusa_ai_chatbot\nodes tests -g "*.py" -g "*.md"
rg -n "l2d_action_initializer.*l3_|l3_.*l2d_action_initializer" src\kazusa_ai_chatbot\nodes tests -g "*.py"
rg -n "l3_contextual_agent|l2c_judgment_core|l2d_action_initializer|l4_collector" src\kazusa_ai_chatbot\nodes tests -g "*.py" -g "*.md"
```

Allowed static-check residue must be limited to historical execution evidence
inside development-plan text or explicit migration notes. Runtime source and
tests must use the new names.

### Real LLM Verification

Run one case at a time:

```powershell
COGNITION_CONNECTION_CASE_FILE=test_artifacts\cognition_stage_connection\qq_673225019_private_cases.json COGNITION_CONNECTION_CASE_ID=qq_private_001 venv\Scripts\python -m pytest tests\test_cognition_stage_connection_live_llm.py::test_live_cognition_stage_connection_case -q -s -m live_llm
```

Then inspect the trace before running the next case. Repeat for all captured
private and group cases.

Each trace must include:

- case id and source kind;
- historical user message and assistant reply;
- L1 output;
- L2a output;
- L2b output;
- L2c1 output;
- L2c2 social context output;
- L2d prompt payload and selected action specs;
- selected L3 text outputs when `speak` is selected;
- L4 collected directives when `speak` is selected;
- dialog output when `speak` is selected;
- comparison report against the historical assistant reply;
- manual judgment note.

## Independent Code Review

Before marking this plan complete, run an independent code review pass over the
full diff. The reviewer must verify:

- no LLM responsibility changed;
- contextual/social appraisal moved to L2c2 and still emits only the same four
  social fields;
- L2d remains the only LLM action selector;
- selected L3 no longer calls contextual/social appraisal;
- L4 still receives all social, linguistic, and visual inputs;
- no old response gate was reintroduced;
- real LLM evidence covers both private and group QQ cases for user
  `673225019`;
- docs and tests use the new stage names;
- verification commands and trace paths in `Execution Evidence` are accurate.

Review findings inside the approved change surface may be fixed directly.
Findings that require new action kinds, new prompt responsibilities, public API
changes, or data migration require plan revision before code changes.

## Acceptance Criteria

- Internal stage names match the target naming in source, tests, and docs.
- `L2c2 social_context_appraisal` runs before `L2d action_selection`.
- `L2d action_selection` receives prompt-safe social context.
- L2d output contract remains semantic `action_requests` materialized by
  deterministic code.
- Selected L3 text does not call contextual/social appraisal.
- L3 style/content/preference/visual and L4 collector run only when `speak` is
  selected.
- No selected `speak` skips L3, L4, and dialog while still producing private
  trace evidence when applicable.
- Real LLM integration traces exist and are inspected for QQ user `673225019`
  private and group cases.
- No prompt or code path reintroduces response-decision authority outside L2d.
- Independent code review is completed and recorded.

## Risks

| Risk | Mitigation |
|---|---|
| Moving contextual earlier subtly changes prompt behavior | Keep prompt responsibility and output fields unchanged; validate with real LLM private/group traces. |
| L2d prompt becomes overloaded | Add only four already-semantic social fields; do not add raw history, ids, permissions, delivery data, or handler details. |
| Selected L3 loses required social fields | Add deterministic tests that L3 visual and L4 collector receive fields from state. |
| Old stage names survive in prompt-selection tests | Use static grep and update source/tests/docs in the same stage. |
| Live LLM evidence becomes uninspectable | Write per-case trace files with stage outputs and manual judgment notes. |
| Real group data for `673225019` is unavailable | Stop and record a data blocker; do not substitute synthetic group cases for the required real source. |

## Execution Evidence

Record execution evidence here as stages complete. Each entry must include
date, agent, commands, result, trace paths for live LLM cases, and whether the
stage checkbox was signed off.

### 2026-05-16 Draft Creation

- Drafted plan from system-engineering review and user decision:
  no LLM responsibility changes; only graph placement and internal naming
  changes.
- Registered required real LLM evidence source: QQ user `673225019` private and
  group chat history.
