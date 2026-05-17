# self cognition dialog state contract bugfix plan

## Summary

- Goal: Stop self-cognition and dialog graph paths from treating semantic intent labels or marker text as sufficient evidence for visible dialog, and enforce a single runtime dialog-state contract before any dialog LLM node reads `action_directives`.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Execution prerequisite: do not start implementation until `dialog_anchor_authority_stale_history_bugfix_plan.md` is completed or superseded and its changes to `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, `tests/test_dialog_agent.py`, `tests/test_dialog_generator_live_llm_contract.py`, and `tests/test_dialog_anchor_boundary_live_llm.py` are merged or intentionally parked outside this worktree.
- Overall cutover strategy: bigbang route/source-packet/contract hardening with no database migration, no prompt rewrite, no adapter change, and no new LLM call.
- Highest-risk areas: changing legacy self-cognition route behavior, deleting marker-based visible candidates, changing private-finalization consolidation behavior, weakening fail-fast state validation, and accidentally treating missing internal graph state as optional defaults.
- Acceptance criteria: the new RED self-cognition tests pass, dialog contract tests raise `dialog_agent.StateContractError` instead of raw `KeyError`, existing `speak` L2d/L3 handoff still passes, stale marker/private-finalization tests are rewritten, and `run_self_cognition_worker_tick` records per-case state-contract failures without the tick failing.

## Context

The production failure on 2026-05-17 was:

```text
persona_supervisor2_cognition output:
  logical_stance=CONFIRM
  character_intent=PROVIDE
  action_specs=[]

self_cognition.runner selected dialog work anyway
dialog_agent.dialog_generator read:
  state["action_directives"]["linguistic_directives"]

result:
  KeyError: 'linguistic_directives'
```

The root cause is not the `KeyError` itself. The root cause is a contract split:

- Compile-time `TypedDict` state definitions document required graph fields.
- Runtime graph edges accept plain dictionaries from live, dry-run, and background paths.
- Self-cognition still has legacy route heuristics that treat `character_intent` values including `PROVIDE` as outward-contact selection.
- Self-cognition source-packet framing still tells the model to emit `[ACTION_CANDIDATE]`, creating a second visible-output authority outside the newer `speak` action-spec path.
- The newer L2d/L3 contract says visible dialog must be backed by a selected `speak` `action_spec`, with L3 collecting complete `action_directives`.
- Dialog then uses direct indexing as fail-fast validation, but it is too late and too low-level, so the worker logs a raw missing-key crash instead of a boundary contract error or a non-dialog route.

Relevant source boundaries:

- `src/kazusa_ai_chatbot/self_cognition/tracking.py`: selects self-cognition route.
- `src/kazusa_ai_chatbot/self_cognition/projection.py`: renders source-packet agency options.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`: builds dialog state, optionally calls L3, calls dialog, and builds consolidation state.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`: records per-case worker failures without crashing the tick.
- `src/kazusa_ai_chatbot/self_cognition/README.md`: documents the self-cognition ICD and runtime budget.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: consumes `action_directives.linguistic_directives` and `action_directives.contextual_directives`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: collects L3 text-surface directives for selected `speak` actions.
- `tests/test_self_cognition_tracking.py`: deterministic route and handoff coverage.

## Mandatory Skills

- `local-llm-architecture`: load before changing graph, cognition, L3 surface, dialog, or background LLM behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running deterministic or live LLM tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Check `git status --short` before editing.
- Use `venv\Scripts\python` for Python commands.
- Use `apply_patch` for manual edits.
- Do not read `.env`.
- Do not alter user changes outside this plan's change surface.
- Do not start implementation while `dialog_anchor_authority_stale_history_bugfix_plan.md` is `in_progress` or while its dialog files are dirty in this worktree. If the prerequisite is not met, stop before editing code.
- Treat `action_specs` as the deterministic action-selection source for new self-cognition visible output.
- Treat L3 `action_directives` as required internal graph state before dialog generation or evaluation.
- Do not add `.get(..., fallback)` defaults for required internal dialog state.
- Do not add an LLM retry, repair call, compatibility shim, feature flag, adapter fallback, prompt workaround, or model-facing keyword filter.
- Keep LLM semantic judgment in cognition/L2d/L3. Keep route eligibility, state validation, persistence, scheduling, and worker error classification in deterministic code.
- Deterministic tests may run in batches. Live LLM tests are not required for this plan.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.

## Must Do

- Preserve the new RED tests in `tests/test_self_cognition_tracking.py` and make them pass by fixing the route/handoff contract, not by weakening the assertions.
- Add unit tests for a typed dialog-ready state contract that cover missing `action_directives`, missing `linguistic_directives`, missing `contextual_directives`, and a valid complete directive shape.
- Delete the legacy self-cognition route path that turns `character_intent` alone into `ROUTE_ACTION_CANDIDATE`.
- Ensure self-cognition calls dialog only when a selected `speak` action spec exists. Complete text directives are required before dialog, but they are not sufficient to authorize dialog.
- Delete `[ACTION_CANDIDATE]` marker authority from source-packet framing, route classification, action-candidate text extraction, and tests.
- Ensure private-only action specs still route to audit/consolidation without calling dialog.
- Add `StateContractError(ValueError)` in `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, raised before dialog LLM work when required internal fields are absent.
- Add `validate_dialog_action_directives(state: dict[str, Any], *, usage_mode: str) -> tuple[dict[str, Any], dict[str, Any]]` in `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- Use `validate_dialog_action_directives` in `dialog_agent`, `dialog_generator`, `dialog_evaluator`, and `self_cognition.runner._build_dialog_state_with_text_surface`.
- Remove the private-finalization dialog initiation path from `self_cognition.runner._build_consolidation_ready_state`. Consolidation may reuse an already-rendered action-candidate `dialog_output`, but it must not start a fresh dialog call.
- Update the self-cognition ICD in `src/kazusa_ai_chatbot/self_cognition/README.md` so it no longer documents `[ACTION_CANDIDATE]` marker authority or fresh private-finalization dialog calls.
- Add worker handling for `StateContractError` inside `run_self_cognition_worker_tick`: increment `failed_count`, record a runtime error event with `stack_fingerprint="self_cognition_case_state_contract"`, do not increment `processed_count`, and continue to the next case.
- Make raw `KeyError` from missing dialog contract fields unreachable in normal dialog-agent entrypoints and self-cognition dialog calls.
- Preserve fail-fast behavior: missing required internal state must produce a typed contract failure, not empty defaults.
- Update obsolete tests that explicitly assert legacy intent-only visible routing.
- Run all verification commands in this plan and record output in `Execution Evidence`.

## Deferred

- Do not redesign cognition, L2d, L3, dialog prompts, RAG, queue intake, adapter delivery, persistence schemas, memory consolidation, or scheduler policy.
- Do not add compatibility paths for old self-cognition intent-only visible output.
- Do not add default empty `action_directives` at downstream read sites to hide missing upstream state.
- Do not add new LLM calls, retries, repair prompts, or evaluator stages.
- Do not change `persona_supervisor2.py` no-response `_empty_action_directives()` behavior.
- Do not modify live LLM tests.
- Do not address unrelated `KeyError` classes outside dialog-ready graph-state contract boundaries in this plan.
- Do not create a new state-contract module. The validator and exception live in `src/kazusa_ai_chatbot/nodes/dialog_agent.py` for this plan.
- Do not keep `_needs_private_dialog_finalization`; delete it instead of changing its behavior behind the same name.
- Do not keep `[ACTION_CANDIDATE]` as a source-packet, route, or test marker.
- Do not add new worker retry, quarantine, scheduler-failed, or database-status behavior. This plan records contract failures through existing runtime-error and worker-event telemetry only.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Self-cognition route eligibility | bigbang | Replace intent-only action-candidate routing with action-spec/directive-backed eligibility. Do not preserve intent-only compatibility. |
| Marker route authority | bigbang | Delete `[ACTION_CANDIDATE]` marker authority. Do not preserve marker compatibility in source packets, routing, or tests. |
| Dialog state validation | bigbang | Add typed runtime validation before dialog LLM work. Do not allow raw nested `KeyError` for required contract fields. |
| Worker case failures | bigbang | Convert per-case `StateContractError` into failed case telemetry without failing the tick. Do not add DB retry/quarantine behavior. |
| Tests | bigbang | Update legacy expectations to the new contract in the same change. |
| Prompts and LLM calls | bigbang no-op | Do not change prompt text or call count. |
| Database, queue, adapters | bigbang no-op | Make no changes. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a compatible strategy by default.
- For `bigbang` areas, delete or rewrite old references instead of preserving them.
- Any change to this cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent must use the exact exception and validator names specified in `Must Do`: `StateContractError` and `validate_dialog_action_directives`.
- The agent must not create any new helper module, class hierarchy, protocol, dataclass, or compatibility wrapper for state contracts.
- The agent must not choose alternate function names, alternate exception names, or alternate test file locations.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, feature flags, helper agents, or extra features.
- The agent must not change files outside the listed change surface. If verification exposes a required outside edit, stop and report the exact blocker instead of editing outside scope.
- Existing caller dependence on intent-only visible self-cognition does not preserve the legacy behavior. If discovered, stop and report the caller list as a blocker.
- Existing caller dependence on `[ACTION_CANDIDATE]` marker self-cognition does not preserve marker behavior. If discovered, stop and report the caller list as a blocker.
- If a required instruction is impossible without prompt, database, queue, or adapter changes, stop and report the blocker instead of inventing a substitute.

## Target State

Self-cognition visible output follows this deterministic gate:

```text
cognition/L2d output
  -> selected speak action spec exists
  -> L3 text surface collects complete action_directives
  -> dialog receives validated dialog-ready state
  -> dialog renders visible candidate text
```

Non-visible self-cognition follows this gate:

```text
cognition output with no speak action
  -> audit/progress/private-action route
  -> no dialog call
  -> consolidation receives final_dialog=[]
```

Invalid explicit visible requests follow this gate:

```text
explicit action_candidate route without selected speak, or selected speak without complete directives
  -> StateContractError at self-cognition runner/dialog boundary
  -> worker records runtime error and failed_count for that case
  -> no raw KeyError from dialog internals
```

Dialog-agent required state is:

```python
{
    "action_directives": {
        "linguistic_directives": dict,
        "contextual_directives": dict,
    },
    "dialog_usage_mode": str,
}
```

The validator contract is:

```python
class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


def validate_dialog_action_directives(
    state: dict[str, Any],
    *,
    usage_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return linguistic and contextual directives or raise StateContractError."""
```

The validator must raise `StateContractError` with these exact missing-path fragments:

```text
action_directives
action_directives.linguistic_directives
action_directives.contextual_directives
action_specs.speak
```

Other existing dialog fields remain required by their current direct-index reads. This plan only scopes the `action_directives` failure class.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Root source for visible self-cognition | Selected `speak` action spec only | This matches the L2d/L3 split and avoids semantic intent labels or marker text being interpreted as delivery decisions. |
| Intent-only fallback | Delete `_cognition_selects_outward_contact` and delete its `classify_route` call | `PROVIDE` describes semantic stance, not a transport-ready visible action. |
| Content-anchor fallback | Keep only progress/audit/silent marker routes from `_route_from_content_anchors`; delete `[ACTION_CANDIDATE]` and `[ANSWER]` visible-route authority | Content anchors are content, not delivery authorization. |
| Missing dialog state | Raise `dialog_agent.StateContractError` before LLM work | Direct indexing remains fail-fast, but the failure belongs at the component boundary with a clear component/path message. |
| Defaults | Do not default missing nested directives to `{}` | Missing internal graph state is a bug and must not be hidden. |
| Private finalization | Delete fresh private dialog finalization from consolidation path | This alternate path is the synthetic route that let incomplete state reach dialog. Consolidation can consume `final_dialog=[]` or reuse prior visible render output. |
| Validator location | Put `StateContractError` and `validate_dialog_action_directives` in `dialog_agent.py` | The contract is local to dialog consumption and the runner already imports dialog-agent constants. |
| Worker failure handling | Catch `StateContractError` per case in `run_self_cognition_worker_tick`, increment `failed_count`, record runtime error event, continue | This keeps a bad generated state from failing the whole worker tick without adding retry or DB migration scope. |
| Test location | Put new contract tests in `tests/test_dialog_agent.py` | Existing dialog tests and fixtures already live there. |
| Prompts | No prompt changes | The observed failure occurs before prompt quality matters. |
| Live LLM tests | Not required | This is deterministic orchestration and state validation. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - Delete `_cognition_selects_outward_contact`.
  - Delete the `classify_route` branch that calls `_cognition_selects_outward_contact`.
  - Delete `extract_action_candidate_text`.
  - Keep `_route_from_content_anchors`, but ensure it only maps `PROGRESS_MAINTENANCE_MARKER`, `AUDIT_ONLY_MARKER`, and `SILENT_NO_WRITE_MARKER`; do not add `[ACTION_CANDIDATE]` or `[ANSWER]` fallback behavior.
  - Keep explicit route and action-spec route logic deterministic and testable.
- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Delete `ACTION_CANDIDATE_MARKER`.
  - Delete `OUTWARD_CONTACT_INTENTS`.
  - Delete `OUTWARD_CONTACT_STANCES`.
- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - Replace the agency option that tells the model to emit a proactive message candidate with `[ACTION_CANDIDATE]`.
  - New agency option text must instruct that outward contact requires selecting the visible `speak` action through the shared action-spec contract.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Gate dialog calls on selected `speak`; complete directives are required but do not authorize dialog by themselves.
  - Preserve no-dialog consolidation for audit/progress/private-only cases.
  - Import `validate_dialog_action_directives` from `kazusa_ai_chatbot.nodes.dialog_agent`.
  - Add `_validate_self_cognition_dialog_state(dialog_state: dict[str, Any], *, usage_mode: str) -> None` near `_build_dialog_state_with_text_surface`.
  - `_validate_self_cognition_dialog_state` must raise `StateContractError` with a message containing `usage_mode=<value>` and `action_specs.speak` when `_has_selected_speak_action(dialog_state)` is false.
  - `_validate_self_cognition_dialog_state` must call `validate_dialog_action_directives(dialog_state, usage_mode=usage_mode)` after the speak check.
  - In `_build_dialog_state_with_text_surface`, call `_validate_self_cognition_dialog_state(dialog_state, usage_mode=usage_mode)` after the optional L3 surface update and before returning `dialog_state`.
  - Remove the `tracking.extract_action_candidate_text(cognition_output)` path. When `action_attempt["status"] == ACTION_ATTEMPT_STATUS_CANDIDATE`, build dialog state through `_build_dialog_state_with_text_surface` and render through dialog.
  - Delete `_needs_private_dialog_finalization`.
  - In `_build_consolidation_ready_state`, remove the branch that starts a new dialog call when `dialog_output is None`. If `dialog_output is None`, set `active_dialog_output` to the empty final-dialog payload and `dialog_called = False`.
  - Update public function docstrings and argument descriptions that currently say `dialog_client` is for private finalization; describe it as the dialog render seam for selected visible `speak`.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Import `StateContractError` from `kazusa_ai_chatbot.nodes.dialog_agent`.
  - Add `failed_count: int = 0` to `SelfCognitionWorkerResult`.
  - Wrap only the per-case artifact build call in `run_self_cognition_worker_tick` with `except StateContractError as exc`.
  - In that handler, increment `result.failed_count`, call `event_logging.record_runtime_error_event(component="self_cognition.worker", error_class="StateContractError", error_preview=str(exc), stack_fingerprint="self_cognition_case_state_contract", top_frame_module=__name__, recovered=True)`, then `continue`.
  - Do not increment `processed_count` for contract-failed cases.
  - Pass `failed_count=result.failed_count` into `_record_worker_tick_event`.
  - In `_record_worker_tick_event`, set `status="failed"` when `processed_count == 0 and failed_count > 0`; set `status="completed"` when some cases processed and some failed.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Remove documentation that the service worker invokes dialog for private finalization when consolidation is applied.
  - Remove documentation that private finalization text is sufficient to make an episode consolidatable.
  - Update runtime budget to state that consolidation does not add a dialog call; only selected visible `speak` can invoke L3/dialog.
  - Replace future-cognition handoff text that mentions local delivery-candidate artifacts from marker text with selected `speak` action-candidate artifacts.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Import `Any` from `typing`.
  - Add `StateContractError(ValueError)` immediately after dialog usage-mode constants.
  - Add `validate_dialog_action_directives` immediately after `StateContractError`.
  - In `dialog_agent`, call `validate_dialog_action_directives(global_state, usage_mode=usage_mode)` immediately after `usage_mode = _dialog_usage_mode(global_state)`, store the returned `linguistic_directives`, and use it for the final dialog-quality `anchor_count`.
  - In `dialog_generator`, call `validate_dialog_action_directives(state, usage_mode=state["dialog_usage_mode"])` before building `msg`, and use the returned directive dictionaries in `msg`.
  - In `dialog_evaluator`, call `validate_dialog_action_directives(state, usage_mode=state["dialog_usage_mode"])` before building `msg`, and use the returned directive dictionaries in `msg`.
  - The raised message must include `usage_mode=<value>` and the missing path fragment.
- `tests/test_self_cognition_tracking.py`
  - Keep the new RED tests and update legacy expectations that contradict the new route contract.
  - Replace `_action_cognition_output` so it returns a selected `_speak_action_spec()` instead of `[ACTION_CANDIDATE]` marker text.
  - Rename `test_classify_route_returns_action_candidate_when_cognition_intent_provides` to `test_classify_route_does_not_treat_answer_anchor_as_contact_without_speak` and change its assertion to `models.ROUTE_AUDIT_ONLY`.
  - Update `test_contact_decision_without_candidate_marker_uses_dialog_candidate` so the cognition fixture includes `_speak_action_spec()` and patch `call_l3_text_surface_handler` to return `_surface_action_directives()`. Do not let this test call the real L3 handler.
  - Update `test_runner_apply_consolidation_builds_private_finalization_state` to assert consolidation no longer starts a private dialog call when no action-candidate dialog was already rendered.
  - Add `test_runner_raises_state_contract_error_for_explicit_action_candidate_without_speak`.
- `tests/test_dialog_agent.py`
  - Import `StateContractError` and `validate_dialog_action_directives`.
  - Add `test_validate_dialog_action_directives_requires_action_directives`.
  - Add `test_validate_dialog_action_directives_requires_linguistic_directives`.
  - Add `test_validate_dialog_action_directives_requires_contextual_directives`.
  - Add `test_validate_dialog_action_directives_accepts_complete_directives`.
  - Add `test_dialog_agent_validates_action_directives_before_llm_call`.
- `tests/test_self_cognition_framing.py`
  - Update `test_self_cognition_framing_presents_agency_without_silence_bias` to assert `[ACTION_CANDIDATE]` is absent and the rendered agency options mention visible `speak`.
- `tests/test_self_cognition_integration.py`
  - Update private-finalization worker tests so no default-path case expects `dialog_usage_mode == "self_cognition_private_finalization"` or private finalization text.
  - Add `test_worker_tick_records_state_contract_error_without_tick_failure`.
  - The new worker test must use an injected `run_case_func` that raises `StateContractError("usage_mode=self_cognition_action_candidate_render missing action_specs.speak")`, patch `worker.event_logging.record_runtime_error_event`, and assert `processed_count == 0`, `failed_count == 1`, `record_runtime_error_event` awaited once, and `record_worker_event` records `failed_count=1`.
- `tests/test_self_cognition_event_logging.py`
  - Update event-log fixture budgets and sanitized payload examples that assume `dialog_calls=1` for private finalization.
  - Add or update assertions so worker events can carry `failed_count` for contract failures without exposing source packet text or dialog text.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Keep L3 surface collection behavior unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Keep no-response empty directive shell unchanged.
- RAG, queue, adapters, persistence, scheduler policy, prompt files, and live LLM regression tests.

### Create

- Do not create new files or modules for this plan.

### Delete

- Delete `self_cognition.tracking._cognition_selects_outward_contact`.
- Delete `self_cognition.tracking.extract_action_candidate_text`.
- Delete `self_cognition.models.ACTION_CANDIDATE_MARKER`.
- Delete `self_cognition.models.OUTWARD_CONTACT_INTENTS`.
- Delete `self_cognition.models.OUTWARD_CONTACT_STANCES`.
- Delete `self_cognition.runner._needs_private_dialog_finalization`.
- Do not delete unrelated tests; rewrite legacy tests named in `Change Surface`.

## Overdesign Guardrail

- Actual problem: background self-cognition can route intent-only or marker-only cognition output into dialog with missing L3 directives, producing raw `KeyError` crashes.
- Minimal change: make visible dialog eligibility depend on selected `speak`, validate required dialog state at the runner/dialog boundary, and record per-case contract failures in the worker.
- Ownership boundaries: cognition/L2d owns semantic/action selection; L3 owns text-surface directives; deterministic projection/tracking/runner owns route eligibility and state validation; dialog owns final wording after validated directives; worker owns per-case failure telemetry.
- Rejected complexity: no prompt repair, no LLM retry, no compatibility shim for intent-only or marker-only output, no feature flag, no adapter fallback, no broad graph-state framework, no default empty directives, no live LLM test expansion, no DB retry/quarantine state.
- Evidence threshold: add broader graph-state framework only after a second component boundary needs the same nontrivial typed validation pattern and focused tests prove local validation is duplicating logic.

## LLM Call And Context Budget

- Before: zero additional LLM calls; self-cognition may call cognition, L3 only when selected `speak` needs directives, dialog when visible output needs text, and consolidation when requested. Legacy private finalization can add a dialog call when consolidation is applied.
- After: same call count or fewer. Intent-only no-speak output must not call dialog.
- Response path: no change.
- Background path: fewer dialog calls because private finalization and marker-only dialog routing are removed.
- Context budget: no prompt change is authorized. The self-cognition source-packet `agency_options` text changes by less than 500 characters and remains inside the existing source-packet character budget.

## Implementation Order

1. Add dialog state contract unit tests in `tests/test_dialog_agent.py`.
   - Add `test_validate_dialog_action_directives_requires_action_directives`.
   - Add `test_validate_dialog_action_directives_requires_linguistic_directives`.
   - Add `test_validate_dialog_action_directives_requires_contextual_directives`.
   - Add `test_validate_dialog_action_directives_accepts_complete_directives`.
   - Add `test_dialog_agent_validates_action_directives_before_llm_call`.
   - Expected before implementation: collection or import fails because `StateContractError` and `validate_dialog_action_directives` do not exist.
2. Re-run the existing RED self-cognition tests.
   - Command is listed in `Verification`.
   - Expected before implementation: two failures from intent-only route and unwanted dialog call.
3. Implement route cleanup in `src/kazusa_ai_chatbot/self_cognition/tracking.py` and `models.py`.
   - Delete `_cognition_selects_outward_contact`.
   - Delete its `classify_route` call branch.
   - Delete `extract_action_candidate_text`.
   - Delete `ACTION_CANDIDATE_MARKER`, `OUTWARD_CONTACT_INTENTS`, and `OUTWARD_CONTACT_STANCES`.
   - Ensure `PROVIDE` alone with `action_specs=[]` returns `ROUTE_AUDIT_ONLY` for past-due commitment cases.
   - Update the named obsolete tests from `Change Surface`.
4. Update self-cognition source-packet framing in `src/kazusa_ai_chatbot/self_cognition/projection.py`.
   - Replace `[ACTION_CANDIDATE]` agency text with visible `speak` action-spec wording.
   - Update `tests/test_self_cognition_framing.py`.
5. Implement dialog readiness gating in `src/kazusa_ai_chatbot/self_cognition/runner.py`.
   - Ensure no-speak/no-complete-directive states do not call dialog.
   - Ensure invalid explicit action-candidate states fail with typed contract error.
   - Delete fresh private dialog finalization from consolidation.
6. Implement typed dialog contract validation in `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
   - Add `StateContractError`.
   - Add `validate_dialog_action_directives`.
   - Call validation in `dialog_agent`, `dialog_generator`, and `dialog_evaluator` exactly as listed in `Change Surface`.
7. Implement worker per-case contract failure handling in `src/kazusa_ai_chatbot/self_cognition/worker.py`.
   - Add `failed_count`.
   - Catch `StateContractError` around the artifact build call only.
   - Record runtime error event and continue.
8. Update self-cognition ICD and stale private-finalization tests.
   - Update `src/kazusa_ai_chatbot/self_cognition/README.md`.
   - Update named tests in `tests/test_self_cognition_integration.py` and `tests/test_self_cognition_event_logging.py`.
9. Run focused tests.
   - Re-run new dialog contract tests.
   - Re-run new self-cognition RED tests.
   - Re-run positive control speak handoff test.
   - Re-run worker contract failure test.
10. Run adjacent regression tests listed in `Verification`.
11. Perform independent code review and remediate in-scope findings.

## Progress Checklist

- [x] Stage 1 - dialog contract tests added
  - Covers: implementation order step 1.
  - Verify: focused dialog contract test command fails before implementation for the expected missing contract behavior.
  - Evidence: record command output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-17` after verification and evidence were recorded.
- [x] Stage 2 - route cleanup complete
  - Covers: implementation order steps 2-4.
  - Verify: `test_classify_route_does_not_use_intent_label_without_speak_or_anchor` passes.
  - Evidence: record changed route/projection symbols and focused test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-17` after verification and evidence were recorded.
- [x] Stage 3 - runner dialog gate complete
  - Covers: implementation order step 5.
  - Verify: `test_runner_does_not_call_dialog_for_intent_only_no_speak` and `test_runner_skips_dialog_for_private_only_actions_without_directives` pass.
  - Evidence: record command output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-17` after verification and evidence were recorded.
- [x] Stage 4 - dialog boundary validation complete
  - Covers: implementation order step 6.
  - Verify: dialog contract unit tests pass and no raw `KeyError` is expected for missing directive paths.
  - Evidence: record exception class and test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-17` after verification and evidence were recorded.
- [x] Stage 5 - worker and ICD updates complete
  - Covers: implementation order steps 7-8.
  - Verify: worker contract-failure test and self-cognition framing test pass.
  - Evidence: record command output and docs touched.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-17` after verification and evidence were recorded.
- [x] Stage 6 - focused and adjacent verification complete
  - Covers: implementation order steps 9-10.
  - Verify: every command in `Verification` passes or has an explicitly recorded unrelated failure.
  - Evidence: record command outputs.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `Codex/2026-05-17` after verification and evidence were recorded.
- [x] Stage 7 - independent code review complete
  - Covers: implementation order step 11.
  - Verify: independent review gate completed and any in-scope findings fixed with affected tests rerun.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual risks, and approval status.
  - Handoff: plan may be marked completed only after this is checked.
  - Sign-off: `Codex/2026-05-17` after review evidence was recorded.

## Verification

### Focused RED/PASS Tests

- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_classify_route_does_not_use_intent_label_without_speak_or_anchor tests\test_self_cognition_tracking.py::test_runner_does_not_call_dialog_for_intent_only_no_speak tests\test_self_cognition_tracking.py::test_runner_skips_dialog_for_private_only_actions_without_directives -q`
  - Before implementation: expected `2 failed, 1 passed`.
  - After implementation: expected `3 passed`.
- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_selected_speak_self_cognition_runs_l3_before_dialog -q`
  - Expected before and after implementation: `1 passed`.
- `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py::test_self_cognition_framing_presents_agency_without_silence_bias -q`
  - After implementation: expected pass, with `[ACTION_CANDIDATE]` absent and visible `speak` action-spec wording present.
- `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_worker_tick_records_state_contract_error_without_tick_failure -q`
  - Before implementation: expected collection/import failure or missing `failed_count` failure.
  - After implementation: expected pass with `failed_count=1` and runtime-error telemetry recorded.

### New Dialog Contract Tests

- `venv\Scripts\python -m pytest tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_action_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_linguistic_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_contextual_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_accepts_complete_directives tests\test_dialog_agent.py::test_dialog_agent_validates_action_directives_before_llm_call -q`
  - Before implementation: expected collection/import failure because `StateContractError` and `validate_dialog_action_directives` do not exist.
  - After implementation: expected pass with `StateContractError`.

### Adjacent Deterministic Regression

- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
- `venv\Scripts\python -m pytest tests\test_dialog_agent.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_event_logging.py -q`
- `venv\Scripts\python -m pytest tests\test_l2d_l3_surface_handoff.py -q`

### Static Greps

- `rg -n "ACTION_CANDIDATE_MARKER|\\[ACTION_CANDIDATE\\]|extract_action_candidate_text|OUTWARD_CONTACT_INTENTS|OUTWARD_CONTACT_STANCES|_cognition_selects_outward_contact" src/kazusa_ai_chatbot/self_cognition tests/test_self_cognition*.py`
  - Expected after implementation: no matches. This proves marker authority, the marker extractor, and legacy outward-contact constants are removed from self-cognition source and tests.
- `rg -n "private finalization|self_cognition_private_finalization|DIALOG_USAGE_MODE_SELF_COGNITION_PRIVATE_FINALIZATION" src/kazusa_ai_chatbot/self_cognition tests/test_self_cognition*.py`
  - Expected after implementation: no self-cognition runner, worker, ICD, or self-cognition tests reference private finalization dialog behavior.
- `rg -n "action_directives\"\\]\\[\"linguistic_directives|action_directives\"\\]\\[\"contextual_directives" src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Expected after implementation: direct reads may remain only after the new validator runs at the same entrypoint, and review must confirm raw missing-key exceptions are unreachable.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Prefer a reviewer that did not implement the change. If no separate reviewer is available, the active agent must reread this plan, inspect the full diff from a fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for changed Python, tests, and plan artifacts.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, `Implementation Order`, `Verification`, and `Acceptance Criteria`.
- Code quality and design weaknesses, including hidden fallback paths, compatibility shims, default empty directives, broad exception handling, duplicated validators, prompt/RAG leaks, and avoidable blast radius.
- Regression quality, including whether tests prove both route cleanup and dialog boundary validation, and whether existing `speak` handoff behavior remains intact.

Fix concrete findings directly only when the fix is inside the approved change surface. If a finding requires prompt, database, adapter, or scheduler changes, stop and update the plan or request approval before implementation.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `PROVIDE` or `CONFIRM` without selected `speak` does not select or execute visible dialog.
- `[ACTION_CANDIDATE]` marker authority is absent from self-cognition source packet framing, route classification, tests, and models.
- Intent-only self-cognition with `action_specs=[]` consolidates without dialog and records `dialog_calls=0`.
- Private-only action specs consolidate without dialog and record `dialog_calls=0`.
- Selected `speak` action specs still run L3 before dialog.
- Missing `action_directives.linguistic_directives` or `action_directives.contextual_directives` fails with the approved typed contract error before dialog LLM work.
- Missing selected `speak` for a self-cognition visible dialog path fails with `StateContractError` containing `action_specs.speak`.
- Worker ticks record per-case `StateContractError` failures with `failed_count` and do not fail the whole tick.
- The self-cognition ICD no longer documents private-finalization dialog calls or marker-based action candidates.
- Focused and adjacent deterministic verification commands pass.
- Independent code review is completed and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Legacy tests relied on intent-only visible routing | Update those tests to the new action-spec/directive contract rather than preserving the legacy path | Full `tests/test_self_cognition_tracking.py` run |
| Legacy tests relied on `[ACTION_CANDIDATE]` marker text | Delete marker authority and rewrite tests to selected `speak` or no-dialog assertions | Static marker grep and `tests/test_self_cognition_tracking.py` |
| Private-finalization deletion leaves stale docs/tests | Update ICD, integration tests, and event logging tests in the same plan | Static private-finalization grep and self-cognition regression tests |
| Typed validation becomes a defaulting shim | Require raised typed errors and forbid empty directive fallbacks | Dialog contract missing-field tests |
| Route cleanup suppresses real selected visible output | Keep selected `speak` as the only self-cognition visible-dialog authority | Existing `test_selected_speak_self_cognition_runs_l3_before_dialog` and added route tests |
| Worker hides contract bugs completely | Record runtime-error telemetry and failed worker count for each contract-failed case | Worker contract failure test |
| Validation misses evaluator/event logging path | Validate before all dialog-agent reads of required directive paths | Static grep and `tests/test_dialog_agent.py` |

## Independent Plan Review

Review performed on 2026-05-17 from a fresh-review posture against
`development-plan-writing`, `plan_contract.md`, `execution_gates.md`, current
source, current tests, and `development_plans/README.md`.

Resolved blockers:

- Active dialog-anchor work conflict is resolved by an explicit execution
  prerequisite and registry execution note. This plan is approved but must not
  be implemented until the prerequisite is cleared.
- Worker-level acceptance is resolved by adding `self_cognition/worker.py`,
  `failed_count`, runtime-error telemetry, and a worker contract-failure test
  to the approved change surface.
- `[ACTION_CANDIDATE]` marker authority is resolved by deleting marker
  authority from models, projection, tracking, runner behavior, tests, and
  static greps.
- Private-finalization deletion is resolved by adding the self-cognition ICD,
  integration tests, and event-logging tests to the approved change surface.
- The deterministic L3 test gap is resolved by requiring
  `test_contact_decision_without_candidate_marker_uses_dialog_candidate` to
  patch `call_l3_text_surface_handler` and return `_surface_action_directives()`.

Approval status: approved for execution only after the execution prerequisite
in `Summary` and `Mandatory Rules` is satisfied.

## Execution Evidence

- 2026-05-17 RED focused command:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_classify_route_does_not_use_intent_label_without_speak_or_anchor tests\test_self_cognition_tracking.py::test_runner_does_not_call_dialog_for_intent_only_no_speak tests\test_self_cognition_tracking.py::test_runner_skips_dialog_for_private_only_actions_without_directives -q
```

Result: `2 failed, 1 passed`.

Failure evidence:

- `test_classify_route_does_not_use_intent_label_without_speak_or_anchor`: expected `audit_only`, actual `action_candidate`.
- `test_runner_does_not_call_dialog_for_intent_only_no_speak`: runner called dialog and triggered the test sentinel `AssertionError("intent-only cognition should not call dialog")`.
- `test_runner_skips_dialog_for_private_only_actions_without_directives`: passed, proving private-only action specs already avoid dialog.

- 2026-05-17 positive control command:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_selected_speak_self_cognition_runs_l3_before_dialog -q
```

Result: `1 passed`, proving the intended selected-`speak` L3-to-dialog handoff already works.

- 2026-05-17 execution prerequisite check:

  - `git status --short --untracked-files=all -- src/kazusa_ai_chatbot/nodes/dialog_agent.py tests/test_dialog_agent.py tests/test_dialog_generator_live_llm_contract.py tests/test_dialog_anchor_boundary_live_llm.py development_plans/active/bugfix/dialog_anchor_authority_stale_history_bugfix_plan.md` returned no dirty prerequisite files.
  - `dialog_anchor_authority_stale_history_bugfix_plan.md` still had a stale `Status: in_progress` header and registry row, but all progress stages in that plan were checked and its scoped files were clean. The user's 2026-05-17 execution instruction was treated as approval to proceed with this evidence as the prerequisite-cleared state. Execution proceeded without touching anchor-plan files.

- 2026-05-17 production-code handoff:

  - Worker agent `019e33ff-7d09-7423-a47e-6c7fea804e89` completed with `DONE_WITH_CONCERNS`.
  - Changed production files: `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, `src/kazusa_ai_chatbot/self_cognition/models.py`, `src/kazusa_ai_chatbot/self_cognition/projection.py`, `src/kazusa_ai_chatbot/self_cognition/tracking.py`, `src/kazusa_ai_chatbot/self_cognition/runner.py`, and `src/kazusa_ai_chatbot/self_cognition/worker.py`.
  - Concern was parent-owned docs cleanup for `src/kazusa_ai_chatbot/self_cognition/README.md`; completed in the parent execution pass.

- 2026-05-17 RED dialog contract command:

```powershell
venv\Scripts\python -m pytest tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_action_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_linguistic_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_contextual_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_accepts_complete_directives tests\test_dialog_agent.py::test_dialog_agent_validates_action_directives_before_llm_call -q
```

Result before implementation: collection/import failure because
`StateContractError` was not yet exported from
`kazusa_ai_chatbot.nodes.dialog_agent`.

- 2026-05-17 RED worker contract command:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_worker_tick_records_state_contract_error_without_tick_failure -q
```

Result before implementation: collection/import failure because
`StateContractError` was not yet exported from
`kazusa_ai_chatbot.nodes.dialog_agent`.

- 2026-05-17 focused PASS commands:

```powershell
venv\Scripts\python -m pytest tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_action_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_linguistic_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_requires_contextual_directives tests\test_dialog_agent.py::test_validate_dialog_action_directives_accepts_complete_directives tests\test_dialog_agent.py::test_dialog_agent_validates_action_directives_before_llm_call -q
```

Result: `5 passed`.

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_classify_route_does_not_use_intent_label_without_speak_or_anchor tests\test_self_cognition_tracking.py::test_runner_does_not_call_dialog_for_intent_only_no_speak tests\test_self_cognition_tracking.py::test_runner_skips_dialog_for_private_only_actions_without_directives tests\test_self_cognition_tracking.py::test_runner_rejects_explicit_visible_route_without_speak -q
```

Result: `4 passed`.

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_selected_speak_self_cognition_runs_l3_before_dialog tests\test_self_cognition_framing.py::test_self_cognition_framing_presents_agency_without_silence_bias tests\test_self_cognition_integration.py::test_worker_tick_records_state_contract_error_without_tick_failure -q
```

Result: `3 passed`.

- 2026-05-17 adjacent deterministic PASS commands:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q
```

Result: `40 passed`.

```powershell
venv\Scripts\python -m pytest tests\test_dialog_agent.py -q
```

Result: `20 passed`.

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py tests\test_l2d_l3_surface_handoff.py -q
```

Result: `28 passed`.

- 2026-05-17 static and compile checks:

```powershell
rg -n "ACTION_CANDIDATE_MARKER|\[ACTION_CANDIDATE\]|extract_action_candidate_text|OUTWARD_CONTACT_INTENTS|OUTWARD_CONTACT_STANCES|_cognition_selects_outward_contact" src/kazusa_ai_chatbot/self_cognition tests --glob "test_self_cognition*.py"
```

Result: no matches.

```powershell
rg -n "private finalization|self_cognition_private_finalization|DIALOG_USAGE_MODE_SELF_COGNITION_PRIVATE_FINALIZATION" src/kazusa_ai_chatbot/self_cognition tests --glob "test_self_cognition*.py"
```

Result: no matches.

```powershell
rg -n 'action_directives"\]\["linguistic_directives|action_directives"\]\["contextual_directives' src/kazusa_ai_chatbot/nodes/dialog_agent.py
```

Result: no matches.

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\worker.py
```

Result: passed.

```powershell
git diff --check
```

Result: exit code `0`; output contained CRLF normalization warnings only.

- 2026-05-17 independent code review:

  - Reviewer agent `019e340a-bfb2-7b52-ad47-6f880c28394d` performed read-only review.
  - Initial findings: plan lifecycle/checklist/handoff inconsistency, prerequisite exception wording, stale self-cognition private-finalization docstrings, and a test-side reconstructed retired marker.
  - Fixes: updated plan lifecycle evidence and checklist, recorded the user-approved prerequisite override, removed stale self-cognition private-finalization docstrings, removed the reconstructed marker test string, and tightened the deterministic L3 seam test.
  - Re-review finding: one stale `private_finalization` phrase remained in a self-cognition test name.
  - Final fix: renamed the test to `test_runner_apply_consolidation_uses_empty_dialog_without_render`.
  - Final reviewer status: approved with no findings. Residual risk: full repository suite was not run.

- 2026-05-17 post-review reruns:

```powershell
rg -n "[Pp]rivate[_ -]finalization|self_cognition_private_finalization|DIALOG_USAGE_MODE_SELF_COGNITION_PRIVATE_FINALIZATION" src/kazusa_ai_chatbot/self_cognition tests --glob "test_self_cognition*.py"
```

Result: no matches.

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py::test_runner_apply_consolidation_uses_empty_dialog_without_render -q
```

Result: `1 passed`.

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q
```

Result: `40 passed`.

```powershell
rg -n "ACTION_CANDIDATE_MARKER|\[ACTION_CANDIDATE\]|extract_action_candidate_text|OUTWARD_CONTACT_INTENTS|OUTWARD_CONTACT_STANCES|_cognition_selects_outward_contact" src/kazusa_ai_chatbot/self_cognition tests --glob "test_self_cognition*.py"
```

Result: no matches.

```powershell
git diff --check
```

Result: exit code `0`; output contained CRLF normalization warnings only.

## Execution Closure

Execution completed on 2026-05-17. The plan is moved to completed history after review approval and recorded verification.
