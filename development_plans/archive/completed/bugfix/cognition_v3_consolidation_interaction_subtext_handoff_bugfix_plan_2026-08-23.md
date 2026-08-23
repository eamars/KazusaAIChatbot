# Cognition V3 Consolidation Interaction-Subtext Handoff Bugfix

## Summary

- Goal: close the live post-turn `KeyError: 'interaction_subtext'` and prove
  the complete debug-chat workflow through background consolidation completes
  without an exception.
- Status: completed
- Scope boundary: the canonical Cognition V3 output projection into the
  existing `GlobalPersonaState` consolidation bridge, its deterministic owner
  test and impact mapping, canonical conformance repairs for three stale
  direct tool-result fixtures exposed by adjacent verification, and one fresh
  live end-to-end verification run.
- Change direction: project the validated
  `active_character_goal.reason` string verbatim as `interaction_subtext`
  while preserving `private_monologue` as the separate
  `internal_monologue` value.
- Acceptance state: completed with deterministic verification, a fresh live
  service run, retained evidence, and independent parent sign-off.

## Scope And Change Direction

The live Cognition V3 path already projects `emotional_appraisal`,
`character_intent`, and `logical_stance` for downstream consolidation, but it
omits the sibling `interaction_subtext` field. Consolidation correctly treats
that field as required internal state and plain-indexes it. The resulting
failure appears only after visible dialog because consolidation runs in the
post-turn background path.

The existing failure is reproduced in:

- `test_artifacts/live_llm/cognition_subjective_continuity_2026-08-23/service.log`
  at the background-consolidation traceback; and
- `test_artifacts/reviews/cognition_subjective_continuity_2026-08-23.md`
  under `Operational Observation`.

The fix belongs to the cognition connector that owns the canonical output to
global-state projection. The consolidator, background runner, service wrapper,
and model contracts retain their existing responsibilities.

## Confirmed Decisions

- The user approved execution and explicitly included the formerly
  out-of-scope consolidation observation.
- The responsible code-and-test executor is fixed to one `gpt-5.6-luna`
  agent with `reasoning_effort=max` at the standard normal-speed runtime.
  This is a plan-scoped fixed execution constraint; only the user may change
  it.
- `active_character_goal.reason` is the canonical bounded semantic reason for
  the selected current interaction goal and is projected verbatim. No
  deterministic synthesis, concatenation, keyword interpretation, or semantic
  fallback is permitted.
- `private_monologue` remains the sole source of `internal_monologue`; it is
  not reused as `interaction_subtext`.
- Live verification keeps `no_remember=false` so the actual consolidation path
  executes and waits for its background completion evidence.

## Mandatory Skills

- `development-plan`: plan execution, evidence, review, and lifecycle closure.
- `local-llm-architecture`: preserve semantic ownership and the smallest
  canonical contract.
- `py-style`: apply before Python changes and during code review.
- `test-style-and-execution`: govern deterministic and live test execution.
- `character-test`: govern the real debug-channel request, response, and fresh
  post-turn log evidence.
- `debug-llm`: require an agent-authored human-readable review of the live run.
- `cjk-safety`: protect the existing CJK test content and require an immediate
  syntax check after editing `tests/test_msg_decontextualizer.py`.

## Mandatory Rules

- Preserve the dirty shared worktree and all concurrent standalone-resolver,
  `llm_interface`, `pyproject.toml`, `resolver_skills`, and agentic-resolver
  changes.
- Keep required internal state fail-fast. Do not add `.get(..., fallback)`, a
  catch-and-ignore branch, a compatibility alias, or a synthetic empty value.
- Keep prompts, Cognition V3 schemas, consolidation schemas, persistence
  policy, and LLM routing unchanged.
- Repair stale test inputs with the canonical
  `build_goal_continuation_ref(...)` builder. Keep the production episode
  validator and its required continuation-lineage contract unchanged.
- Use `venv\Scripts\python` for Python and pytest commands.
- Use `apply_patch` for manual file edits.
- Do not read `.env`; normal service startup may load its configured runtime
  environment through existing code.
- Use exactly one Luna executor and do not spawn further agents.

## Must Do

1. Preserve the captured failure as the before-state evidence.
2. Add the exact required projection in
   `_project_output_to_global_state(...)`.
3. Add a direct deterministic unit test whose goal reason and private
   monologue differ, proving each reaches its separate owned field.
4. Register the new exact test node in the source-test impact manifest.
5. Repair the three stale direct `tool_result_ready.v1` fixtures discovered by
   adjacent verification so they supply a canonical `goal_continuation_ref`
   and reach their intended assertions.
6. Collect and run the exact mapped nodes, then run the complete affected
   fixture files plus adjacent consolidation and service background tests
   proportionate to this boundary.
7. Start an isolated debug brain service, send one private message with
   remembering enabled, retain the raw request and response, wait for post-turn
   progress/residue/consolidation work, and retain the fresh service-log slice.
8. Require positive successful-consolidation evidence and zero occurrences of
   `ERROR`, `Traceback`, `Exception`, or `Background consolidation failed` in
   the fresh end-to-end log slice.
9. Author a readable Markdown live-run review from the retained raw evidence,
   stop the isolated service, and verify its port is clear.

## Deferred

- Prompt or model-quality changes.
- Cognition V3 output-schema expansion.
- Consolidation prompt, lane, target, persistence, or schema redesign.
- Changes to background exception containment.
- Cleanup of historical V2 terminology outside the touched test.
- Concurrent standalone-resolver and LLM-interface work.

## Target State

For every validated Cognition V3 result accepted by
`_project_output_to_global_state(...)`:

```text
global_state.internal_monologue
  == cognition_output.private_monologue

global_state.interaction_subtext
  == cognition_output.active_character_goal.reason
```

Both values remain exact strings supplied by the validated cognition output.
The consolidation builder continues to plain-index `interaction_subtext` and
normalize it only for its existing `subjective_appraisals` list. A missing
canonical goal reason remains an upstream contract failure rather than a
post-turn fallback.

## Execution Roles

### Implementation And Verification Owner

- Responsibility: implement the exact projection, add and register the
  regression test, run deterministic checks, execute the live workflow, and
  produce raw and readable evidence.
- Owned surface:
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py`
  - the existing `persona_supervisor2_cognition.py` row in
    `tests/ownership/source_test_impact_manifest.json`
  - `tests/test_consolidation_origin_metadata.py`
  - `tests/test_msg_decontextualizer.py`
  - a new dedicated directory under `test_artifacts/live_llm/`
  - one new review under `test_artifacts/reviews/`
- Authority: make only the fixed implementation, owner-test, impact mapping,
  and canonical fixture changes above; run the required local service and
  verification commands; create test evidence.
- Applicable skills: all skills listed in `Mandatory Skills`.
- Capability floor: production Python, typed state contracts, pytest impact
  mapping, Windows process control, live debug service testing, and log review.
- Independence requirement: none for implementation and verification.
- Fixed executor: one `gpt-5.6-luna`, `reasoning_effort=max`, standard
  normal-speed runtime.
- Acceptance output: scoped diff, exact command results, raw live artifacts,
  readable review, stopped service, and a concise residual-risk statement.
- Gate: starts only after baseline hashes and owned files are recorded; exits
  only when all deterministic and live acceptance checks pass.

### Independent Sign-Off Owner

- Responsibility: review the Luna diff and evidence against this plan and
  decide closure.
- Owned surface: read-only review of the complete scoped diff, test outputs,
  live artifacts, current status, and plan evidence; lifecycle-document edits
  for final closure only.
- Authority: report findings, fail or pass sign-off, and close/archive the plan
  after every required finding is resolved.
- Applicable skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `character-test`, and `debug-llm`.
- Capability floor: independent production-code, contract, test, and live-log
  review.
- Independence requirement: must not be the Luna implementation executor.
- Acceptance output: explicit finding disposition and lifecycle closeout.
- Gate: starts after the implementation owner returns complete evidence; exits
  only with no unresolved required finding.

## Test Impact And Traceability

| Path | Changed symbol or contract | Semantic owner | Exact deterministic pytest node | Supplemental integration or live check | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | `_project_output_to_global_state` required `interaction_subtext` projection | Cognition connector | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_global_projection_supplies_consolidation_interaction_subtext` | Fresh debug `/chat` through successful background consolidation | deterministic unit + live service | Valid V3 turns cannot reach consolidation without the required subjective interaction reason. |
| `tests/ownership/source_test_impact_manifest.json` | exact source-owner node registration | Test ownership manifest | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary` | `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run` | deterministic governance | The production change cannot bypass exact-node collection and execution. |
| `tests/test_consolidation_origin_metadata.py` | stale completed-tool-result fixture continuation lineage | Consolidation-origin test fixture | `tests/test_consolidation_origin_metadata.py::test_tool_result_origin_supports_completed_result_source`; `tests/test_consolidation_origin_metadata.py::test_origin_builders_reject_wrong_source` | Full `tests/test_consolidation_origin_metadata.py` | deterministic integration | Canonical tool-result fixtures reach the origin behavior under test instead of failing during episode construction. |
| `tests/test_msg_decontextualizer.py` | stale accepted-task-result fixture continuation lineage | Decontextualizer test fixture | `tests/test_msg_decontextualizer.py::test_decontextualizer_leaves_accepted_task_episode_source_owned` | Full `tests/test_msg_decontextualizer.py` plus UTF-8 syntax compilation | deterministic integration | The accepted-task fixture reaches the source-ownership assertion under the current canonical episode contract. |

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: add the
  exact required bridge field beside its existing semantic consolidation
  projections.
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py`: add the
  direct ownership regression.
- `tests/ownership/source_test_impact_manifest.json`: add the new exact unit
  node to the existing source row without disturbing concurrent mappings.
- `tests/test_consolidation_origin_metadata.py`: add one canonical continuation
  reference to the shared completed-tool-result fixture used by the two stale
  tests.
- `tests/test_msg_decontextualizer.py`: add one canonical continuation
  reference to the stale inline accepted-task-result fixture while preserving
  all existing CJK bytes and test semantics.
- This plan and `development_plans/README.md`: parent-owned execution evidence
  and lifecycle registry updates.

### Create

- Dedicated raw live-run artifacts under `test_artifacts/live_llm/`.
- One agent-authored live-run review under `test_artifacts/reviews/`.

### Keep

- `src/kazusa_ai_chatbot/consolidation/core.py` and its required plain-index
  contract.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py` and the service wrapper.
- Cognition prompts, schemas, model calls, consolidation prompts, and durable
  write policy.
- All concurrent work outside the exact modified surface.

## Agent Autonomy Boundaries

The implementation owner may choose local assertion wording, command order,
artifact names, an unused isolated port, and the benign private debug message.
The owner must preserve the exact target mapping and owned files. The fixture
repair must use `build_goal_continuation_ref(...)` with deterministic test-only
identity values and must not alter the assertions those fixtures support. Any
need to change a prompt, schema, production episode validator, consolidation
owner, exception policy, persistence contract, or concurrent resolver file is
a plan conflict and must be returned to the parent without expanding scope.

## Verification

Deterministic verification must include:

- collection and execution of both exact nodes in the traceability table;
- execution of every manifest-mapped required unit node for the changed
  production source;
- adjacent consolidation state and service-background tests that exercise
  consolidation scheduling, the background callable, and successful state
  handoff;
- both formerly failing consolidation-origin nodes and the complete
  `tests/test_consolidation_origin_metadata.py` file;
- the formerly failing decontextualizer node and the complete
  `tests/test_msg_decontextualizer.py` file;
- Python compilation of every changed source and test file, immediately after
  the CJK-containing test edit and again in final verification; and
- `git diff --check` on the scoped change.

Live verification must:

- run one case at a time through the real debug `/chat` endpoint;
- retain request, response, complete fresh log slice, and trace evidence when
  a protected trace id is available;
- wait beyond the response until a successful `Consolidation output` record or
  equivalent positive completion record appears;
- inspect the entire fresh slice for the forbidden exception/error markers;
- record whether progress and residue boundaries completed; and
- stop the service and prove the listener is gone.

The debug-LLM review must distinguish deterministic harness success from the
observed character output and post-turn system behavior.

## Acceptance Criteria

- The exact V3 goal reason reaches `interaction_subtext` without mutation.
- The exact private monologue remains separately projected.
- The source-test impact manifest collects and runs the new owner node.
- All three stale direct tool-result fixtures satisfy the canonical
  continuation-lineage contract and reach their intended assertions.
- All focused and adjacent deterministic checks pass.
- One fresh real debug turn returns a valid response and reaches successful
  background consolidation.
- The fresh live slice contains no `ERROR`, `Traceback`, `Exception`, or
  `Background consolidation failed` record.
- The live artifacts and human-readable review are complete and inspectable.
- The isolated service is stopped and its port is clear.
- Independent parent review finds no unresolved scope, ownership, style,
  verification, or residual-risk issue.

## Progress Checklist

- [x] Captured the prior live failure and traced it to the missing connector
  projection.
- [x] Fixed the architecture decision and exact source/test boundary.
- [x] Recorded pre-handoff baseline hashes and owned file status.
- [x] Completed Luna implementation and deterministic verification.
- [x] Completed Luna live end-to-end verification and review artifact.
- [x] Completed independent parent sign-off.
- [x] Archived the completed plan and updated the lifecycle registry.

## Execution Evidence

### 2026-08-23 Pre-Handoff Baseline

All three implementation-owned files already contain approved or concurrent
work that must be preserved. Their exact pre-handoff state is:

| Path | Status | SHA-256 | Existing diff size versus `HEAD` |
| --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | modified | `DDDAE1500A30F1D14E80955D93959F006607CB12DA71F13F50A53C8F50258B1A` | 7 additions, 1 deletion |
| `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py` | modified | `2E48EFDD8801D94139E6259602F1AF0227790C43EC1CDF8FAD278B7185F7175E` | 53 additions |
| `tests/ownership/source_test_impact_manifest.json` | modified | `AB67FA2FFB0BF9BE6160E1717AB485813968647493C5E5377429F290F0B52BAF` | 331 additions, 193 deletions |

The baseline failure is the repeated post-turn traceback ending at
`consolidation/core.py::_build_consolidator_state` when it plain-indexes
`global_state["interaction_subtext"]`.

### 2026-08-23 Adjacent-Failure Diagnostic And Amended Baseline

The same Luna implementation owner reproduced each adjacent failure as an
individual pytest node with full traceback evidence. The failures occur while
the tests construct `tool_result_ready.v1` episodes, before their intended
assertions:

- `tests/test_consolidation_origin_metadata.py::test_tool_result_origin_supports_completed_result_source`
- `tests/test_consolidation_origin_metadata.py::test_origin_builders_reject_wrong_source`
- `tests/test_msg_decontextualizer.py::test_decontextualizer_leaves_accepted_task_episode_source_owned`

Each raises `CognitiveEpisodeValidationError: tool-result episode requires
goal_continuation_ref`. The canonical requirement entered production in commit
`4cb05aa2` on 2026-08-14; the older direct fixtures were not advanced with that
contract. They do not exercise the new interaction-subtext projection, but
their correction is required for this plan's all-adjacent-checks acceptance
gate and the user's zero-exception end-to-end requirement.

Both newly owned files are clean relative to `HEAD` at the amendment boundary:

| Path | Status | SHA-256 | Existing diff size versus `HEAD` |
| --- | --- | --- | --- |
| `tests/test_consolidation_origin_metadata.py` | clean | `B743DD43FD7F31928A51515FE95EBB7E76354D7734DF6789FA21D9DFD4ED461C` | none |
| `tests/test_msg_decontextualizer.py` | clean | `85C25DB871CA9D2E7AEF38B70626BADD771B4B7FEA62661209249CABB75A5F2C` | none |

### 2026-08-23 Implementation Result

Exactly one plan-scoped executor, `/root/consolidation_handoff_fix`, ran as
`gpt-5.6-luna` with `reasoning_effort=max` at standard normal speed. It made
the following scoped changes:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` now projects
  `output["active_character_goal"]["reason"]` verbatim to
  `interaction_subtext` and preserves `output["private_monologue"]` as
  `internal_monologue`.
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py` proves the
  two different source strings reach their separate destination fields.
- `tests/ownership/source_test_impact_manifest.json` registers the exact new
  projection owner node.
- The two newly owned test files now construct their three tool-result
  fixtures with canonical `build_goal_continuation_ref(...)` values and retain
  their original intended assertions.

Final implementation-owned file hashes are:

| Path | SHA-256 |
| --- | --- |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | `C0E2E26A048553C598484E963418581A13A2541C0CF4CD7A5FDBFCDBD0ADA45A` |
| `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py` | `51BD792FFEE5B573F61CF48074902B3B844422670A4C5354D71DB34F122660C6` |
| `tests/ownership/source_test_impact_manifest.json` | `9A762E802779C44ECB5E860FAAA46F8C696912182C715140FFF28269F2F3EEA1` |
| `tests/test_consolidation_origin_metadata.py` | `23E60E751895581AAB7F2BF771AE332F18FF0EDCF6E69CAD4975FD67D7723D7F` |
| `tests/test_msg_decontextualizer.py` | `44151EFCAB80C630CB05E2E026B54016B90F0CC0D745AC6AA49C5DCF6107A6BF` |

### Deterministic Verification Ledger

The Luna owner executed these commands with the project virtual environment:

```powershell
venv\Scripts\python.exe -c "import ast, pathlib, sys; sys.stdout.reconfigure(encoding='utf-8'); path = pathlib.Path('tests/test_msg_decontextualizer.py'); ast.parse(path.read_text(encoding='utf-8'), filename=str(path)); print(f'AST UTF-8 parse OK: {path}')"
venv\Scripts\python.exe -c "import ast, pathlib, sys; sys.stdout.reconfigure(encoding='utf-8'); path = pathlib.Path('tests/test_consolidation_origin_metadata.py'); ast.parse(path.read_text(encoding='utf-8'), filename=str(path)); print(f'AST UTF-8 parse OK: {path}')"
venv\Scripts\python.exe -m py_compile tests\test_consolidation_origin_metadata.py tests\test_msg_decontextualizer.py
```

Both UTF-8 AST parses and compilation passed with exit code 0.

```powershell
venv\Scripts\python.exe -m pytest -q -ra tests/test_consolidation_origin_metadata.py::test_tool_result_origin_supports_completed_result_source tests/test_consolidation_origin_metadata.py::test_origin_builders_reject_wrong_source tests/test_msg_decontextualizer.py::test_decontextualizer_leaves_accepted_task_episode_source_owned
venv\Scripts\python.exe -m pytest -q -ra tests/test_consolidation_origin_metadata.py tests/test_msg_decontextualizer.py
```

The repaired exact nodes passed 3/3. The complete affected files passed 35/35
(9 plus 26), with no skips and exit code 0.

```powershell
venv\Scripts\python.exe -m pytest --collect-only -q tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_global_projection_supplies_consolidation_interaction_subtext tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary
venv\Scripts\python.exe -m pytest -q -ra tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_global_projection_supplies_consolidation_interaction_subtext tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary
venv\Scripts\python.exe -m scripts.validate_test_impact --base-ref HEAD --run
```

The traceability pair collected 2/2 and passed 2/2. The impact validator
collected 100 exact nodes and completed with 99 passed, one expected Windows
symlink-privilege skip (`WinError 1314`), and exit code 0.

```powershell
venv\Scripts\python.exe -m pytest -q -ra tests/test_consolidator_efficiency.py tests/test_consolidator_origin_policy_db_writer.py tests/test_consolidator_source_aware_payloads.py tests/test_consolidation_module_boundary.py tests/test_consolidation_origin_metadata.py tests/test_consolidation_origin_policy.py tests/test_consolidation_source_policy.py tests/test_consolidation_target_routing.py tests/test_consolidator_group_channel_branch.py tests/test_consolidator_origin_selection.py tests/test_post_turn_lifecycle_record.py tests/test_character_operational_state_consolidation.py
```

The adjacent consolidation batch passed 63/63 with no skips and exit code 0.

```powershell
venv\Scripts\python.exe -m pytest -q -ra tests/test_service_background_consolidation.py::test_post_turn_lifecycle_iterates_after_productive_passes tests/test_service_background_consolidation.py::test_post_turn_lifecycle_skips_structural_blockers tests/test_service_background_consolidation.py::test_build_graph_preserves_consolidation_state_from_supervisor tests/test_service_background_consolidation.py::test_brain_terminal_requires_v2_output_update_and_commit_marker tests/test_service_background_consolidation.py::test_no_remember_skips_consolidation_but_releases_after_other_writes
git diff --check -- tests/test_consolidation_origin_metadata.py tests/test_msg_decontextualizer.py
```

The selected background-service checks passed 5/5. The scoped diff check
returned exit code 0 with only Git line-ending normalization warnings.

### Live End-To-End Evidence

The Luna owner started one isolated real service on port 8017 and sent one
private `POST http://127.0.0.1:8017/chat` request with `no_remember=false`.
The service returned HTTP 200 and a valid character response. Evidence is
retained under
`test_artifacts/live_llm/cognition_v3_consolidation_interaction_subtext_2026-08-23/`
and reviewed in
`test_artifacts/reviews/cognition_v3_consolidation_interaction_subtext_2026-08-23.md`.

- Protected trace: `llmtrace_7a4a7b0b515c4562bccf95c74248fea4`
- Delivery tracking: `151ea4441a75410aa2b774aa437e0463`
- Trace result: 8/8 captured cognition/dialog stages succeeded.
- Post-turn result: one positive `Consolidation output` record at
  `2026-08-23 17:19:52.344 +12:00`.
- Fresh full service slice: `ERROR=0`, `Traceback=0`, `Exception=0`, and
  `Background consolidation failed=0`.
- Progress boundary: one active `conversation_progress.v2` row with
  `turn_count=1`.
- Residue boundary: one valid `internal_monologue_residue.v2` row with both
  residue and operation identifiers.
- Shutdown: tracked service processes stopped and port 8017 was independently
  observed with zero listeners.

Provider JSON-schema retry warnings remain visible in the retained log. They
did not produce an exception, error record, failed stage, or failed workflow.

### Independent Parent Sign-Off

The parent reviewed the exact scoped diffs, canonical ownership boundaries,
test repair semantics, manifest row, raw live artifacts, readable debug-LLM
review, complete fresh log slice, structured persistence evidence, and stopped
listener proof. The review independently counted zero forbidden markers and
one positive consolidation record in the fresh slice.

Finding disposition:

- The initially omitted `interaction_subtext` projection is resolved at its
  owning connector with exact validated source data and no fallback.
- The three adjacent fixture failures were reproduced with full tracebacks,
  traced to an older test-input contract mismatch, repaired canonically, and
  rerun successfully through both focused and full adjacent suites.
- The readable review's initially stale 61/2 result was updated to the final
  63/63 result and now contains no failing acceptance result.
- Concurrent resolver, LLM-interface, configuration, and prior-plan edits were
  preserved outside this plan's owned change surface.

No unresolved scope, ownership, style, deterministic-verification, live-run,
exception, shutdown, or residual-risk finding remains. All acceptance criteria
are satisfied, and this plan is complete.

## Execution Closeout

- Lifecycle state: `completed`; archived under
  `development_plans/archive/completed/bugfix/` and registered in
  `development_plans/README.md`.
- Implementation and verification owner: the single fixed
  `gpt-5.6-luna`, `reasoning_effort=max`, standard normal-speed executor.
- Independent sign-off owner: the parent agent.
- Final disposition: closed with the production end-to-end path returning a
  valid response, completing consolidation, and emitting no forbidden error or
  exception marker in the complete fresh workflow slice.
