# background tool-result delivery current-episode evidence bugfix

## Summary

- Goal: restore visible accepted-task result messages when the result is a
  canonical `tool_result` episode.
- Status: completed
- Scope boundary: Cognition Core V2 evidence-authority projection and the
  deterministic/live regression coverage for accepted-task result delivery.
- Change direction: classify the canonical `tool_result` source as current
  episode evidence everywhere the ordinary goal's relational-willingness
  contract derives authority, while preserving the existing `tool_result`
  source schema and normal cognition/dispatcher ownership.
- Acceptance state: completed after DeepSeek implementation, parent review,
  deterministic verification, individual live-LLM inspection, and closeout.

## Execution Handoff

- Parent ownership: live-LLM/database verification, diff review, scoped
  corrections, evidence updates, and lifecycle closeout.
- DeepSeek implementation ownership:
  `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`,
  `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`,
  `tests/test_accepted_task_prompt_contract.py`,
  `tests/test_cognition_core_v2_relational_willingness.py`, and
  `tests/test_cognition_core_v2_dependencies.py`.
- Parent-owned verification surface:
  `tests/test_cognition_core_v2_relational_willingness_live_llm.py` and
  `test_artifacts/diagnostics/background_tool_result_delivery_live_*.json`.
- Pre-handoff baseline: the owned files were clean relative to `HEAD`; the
  only existing workspace changes were this plan and its registry row.

## Incident Evidence

The production task created at `2026-08-05T09:28:28.521392+00:00` matched:

- job: `job-623f657664da41bf9bcfe41bab768b6d`;
- accepted task: `task-623f657664da41bf9bcfe41bab768b6d`;
- terminal worker result: `needs_user_input`;
- result-delivery attempts: six;
- terminal delivery state: `delivery_failed`;
- visible conversation message ID: empty.

The protected failure capsules show that the result was exposed as evidence
handle `e1` with `source_kind: tool_result`. The current code maps that source
to `contextual_fact_only` and derives the relational validator's
`episode_handles` only from `source_kind: episode`. The model therefore could
not satisfy the required current-episode citation rule, even when it cited
`e1` during repair. Cognition failed before `final_dialog`; dispatcher send
was never reached.

The read-only incident review and raw evidence remain in:

- `test_artifacts/diagnostics/background_task_20260805T092828521392_failure_review.md`;
- `test_artifacts/diagnostics/llm_trace_background_failure_first.json`;
- `test_artifacts/diagnostics/background_work_job_623f6576_latest.json`.

## Confirmed Decisions

- `tool_result` remains the canonical source kind for completed background
  work. The fix must not rewrite it to `episode` or add an alias vocabulary.
- A completed `needs_user_input` task remains a valid result-ready episode;
  the character's normal cognition/dialog path owns the visible clarification.
- The deterministic contract remains strict. The fix supplies the correct
  current evidence authority; it does not default, clamp, rewrite, or infer a
  relational stance.
- The existing three goal-cognition attempts and six delivery attempts remain
  unchanged.
- A deterministic fallback message after result cognition exhaustion is a
  separate reliability decision and is excluded from this bugfix.

## Scope And Change Direction

The current source projection already correctly maps a `tool_result` episode to
`source_kind: tool_result`. The bug is the downstream authority projection:

```text
tool_result episode
  -> e1 / source_kind=tool_result
  -> contextual_fact_only + no ordinary current-episode handle
  -> relational-willingness citation rejection
  -> ordinary_response exhaustion
  -> no final_dialog and no dispatcher send
```

The target path is:

```text
tool_result episode
  -> e1 / source_kind=tool_result
  -> current_episode + current-episode handle e1
  -> valid ordinary_response goal
  -> dialog surface
  -> dispatcher and accepted-task delivery receipt
```

The canonical current-episode source set for this plan is exactly
`episode` and `tool_result`. Other source kinds retain their existing
authority roles unless a separate evidence-backed plan changes them.

## Mandatory Skills

- `development-plan`: governs this plan lifecycle, scope, acceptance, and
  review.
- `local-llm-architecture`: preserves LLM semantic ownership and the bounded
  cognition-to-dialog-to-dispatcher boundary.
- `py-style`: applies to every modified Python production or test file.
- `test-style-and-execution`: separates deterministic contract tests, patched
  handoff tests, and individually inspected real-LLM tests.
- `debug-llm`: applies to the live result-delivery replay and its
  agent-authored Markdown review.
- `llm-trace-debug`: applies when inspecting the protected incident capsule
  and post-fix trace evidence.
- `python-venv`: applies to every Python and pytest command.
- `cjk-safety`: applies if the test fixture or prompt-facing Python changes
  add or refactor CJK string literals.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python checks and tests.
- Keep the source kind, evidence handles, result schemas, task states, and
  dispatcher contracts unchanged except for the authority classification
  explicitly defined here.
- Keep semantic judgment in the LLM and deterministic evidence ownership in
  the contract projection. Do not add keyword routing or post-process a model
  stance.
- Use the canonical `parse_llm_json_output(...)` path already owned by the
  goal stage; add no parser or repair model.
- Run deterministic tests in batches. Run each real-LLM case individually,
  inspect its raw/parsed output and review artifact, then continue.
- Preserve unrelated worktree changes and do not modify the live database or
  manually requeue the incident task as part of implementation verification.

## Must Do

1. Add one canonical current-episode source set in the Cognition Core V2
   contract owner containing `episode` and `tool_result`.
2. Make `project_evidence_provenance_role(...)` return `current_episode` for
   both source kinds while preserving all existing promoted-memory,
   promoted-reflection, conversation, action, resolver, and scheduler roles.
3. Make `run_goal_cognition(...)` derive ordinary-goal current-episode handles
   from that same canonical source set, so `tool_result` handle `e1` reaches
   the initial and repair relational-willingness contracts.
4. Keep `validate_relational_willingness(...)` strict and update its durable
   documentation to describe the current-episode source set rather than
   implying that only literal `episode` rows qualify.
5. Add deterministic regression coverage proving that:
   - a `tool_result` row is model-facing `current_episode` evidence;
   - an ordinary goal citing that row succeeds without structural exhaustion;
   - a history-only relational citation still fails closed;
   - the repair feedback exposes the `tool_result` handle as current-episode
     evidence;
   - the canonical accepted-task source still emits `trigger_source` and
     percept `source_kind` as `tool_result`.
6. Add one individually run real-LLM result-delivery case using the captured
   task-result shape. Require a valid relational-willingness object citing the
   current tool-result evidence, a clarification-oriented goal grounded in the
   missing-information result, and no invented task facts or raw identifiers.
7. Verify the existing patched service handoff and background delivery tests
   still prove that non-empty `final_dialog` reaches dispatcher send and
   accepted-task delivery finalization.
8. Author a human-readable post-fix LLM review from the real artifact before
   declaring the live gate complete.

## Deferred

- Changes to `background_work/delivery.py` retry caps, lease recovery, or
  delivery state transitions.
- A deterministic operational fallback message when cognition fails after
  retries. Such a message would change visible wording ownership and needs a
  separate accepted contract.
- Replaying, requeuing, or mutating the production task found during RCA.
- Changes to the task-resolution worker, coding-agent blocker semantics, or
  the `needs_user_input` result schema.
- Changes to `result_source.py` or `persona_supervisor2_cognition.py` source
  construction; their `tool_result` projection is already canonical and
  tested.
- Prompt rewrites, new LLM stages, retry-count changes, compatibility aliases,
  and fixes for secondary appraisal failures seen in later capsules.
- Broad reclassification of `scheduler_event`, `action_result`, or
  `resolver_observation` without a separate reproduction and plan amendment.
- Full live-database or production smoke verification.

## Target State

For a completed accepted-task result:

- `build_result_ready_episode_from_job(...)` continues to produce a
  `tool_result` episode.
- The current source evidence row has handle `e1`, source kind `tool_result`,
  provenance role `current_episode`, and appears in
  `current_episode_evidence_handles`.
- An ordinary goal may cite `e1` in `relational_willingness.evidence_handles`.
- The validated goal proceeds through the existing facade, dialog, dispatcher,
  conversation persistence, and accepted-task delivery receipt path.
- Genuine cognition or adapter failures continue to use the existing typed
  failure and retry behavior.

## Contracts And Data Shapes

The existing evidence row shape remains unchanged:

```json
{
  "evidence_handle": "e1",
  "evidence_ref": {
    "source_kind": "tool_result",
    "source_id": "task-..."
  },
  "semantic_text": "The task needs additional user-provided information."
}
```

Only its deterministic transient authority role and inclusion in the ordinary
goal's current-episode handle subset change. The six-field
`relational_willingness.v2` object, allowed evidence handles, result source
schema, accepted-task state machine, and delivery receipt fields stay fixed.

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: own the canonical
  current-episode source set, map `tool_result` to `current_episode`, and
  update the validator documentation without relaxing validation.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: consume the
  canonical source set when building ordinary-goal current-episode handles;
  preserve prompt, repair, and validator field names.
- `tests/test_accepted_task_prompt_contract.py`: assert that accepted-task
  evidence remains a typed `tool_result` and receives current-episode
  authority.
- `tests/test_cognition_core_v2_relational_willingness.py`: cover role
  projection, successful ordinary-goal tool-result citation, and history-only
  rejection.
- `tests/test_cognition_core_v2_dependencies.py`: cover initial/repair prompt
  feedback and current-episode handle projection for a `tool_result` row.
- `tests/test_cognition_core_v2_relational_willingness_live_llm.py`: add the
  one production-shaped live result-delivery case and durable raw evidence
  path.

### Create

- `test_artifacts/diagnostics/background_tool_result_delivery_live_review.md`:
  parent-authored human-readable review of the individually inspected live
  result-delivery case.
- `test_artifacts/diagnostics/background_tool_result_delivery_live_*.json`:
  raw live prompt, output, parsed contract, trace metadata, and quality notes.

### Keep

- `src/kazusa_ai_chatbot/background_work/result_source.py`: canonical
  `tool_result` episode construction and prompt-safe result projection.
- `src/kazusa_ai_chatbot/background_work/delivery.py`: delivery claims,
  retries, and accepted-task/job state transitions.
- `src/kazusa_ai_chatbot/service.py`: normal cognition, dialog, and dispatcher
  handoff ordering.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: current
  source construction and `tool_result` semantic projection.
- `tests/test_background_work_delivery.py`: existing patched dispatcher and
  delivery-state coverage; use it in verification without changing the
  production handoff contract.

## Agent Autonomy Boundaries

The implementation agent may choose the local constant name, test helper
placement, and fixture mechanics within the listed files. It may update
docstrings and deterministic assertions required by the canonical source-set
change.

The implementation agent must not rename public evidence source kinds, add
aliases, change result or task schemas, change retry counts, add a fallback
sender, rewrite prompts, or alter the semantic stance in deterministic code.
It must not modify files under `background_work`, `service.py`, dispatcher
adapters, or the task-resolution worker. A required change outside this
surface pauses execution for a plan amendment.

## Verification

1. Capture the pre-change focused baseline and reproduce the current
   `tool_result` contract rejection with a deterministic goal fixture.
2. Run the deterministic contract and goal suites:

   ```powershell
   venv\Scripts\python.exe -m pytest `
     tests\test_accepted_task_prompt_contract.py `
     tests\test_cognition_core_v2_relational_willingness.py `
     tests\test_cognition_core_v2_dependencies.py `
     tests\test_cognition_core_v2_contracts.py `
     tests\test_background_work_delivery.py -q
   ```

3. Run the new real-LLM case individually with `-q -s`, inspect its durable
   artifact, and record the model route, rendered payload, raw output, parsed
   decision, trace status, and human quality judgment. Do not batch it with
   other live cases.
4. Run the adjacent deterministic task-resolution and accepted-task lifecycle
   suites after the focused tests pass.
5. Run `venv\Scripts\python.exe -m compileall -q
   src\kazusa_ai_chatbot\cognition_core_v2` and `git diff --check`.
6. Review the complete diff for source-kind preservation, authority ownership,
   no fallback/retry drift, no schema aliases, and no unrelated changes.

## Acceptance Criteria

1. The deterministic reproduction of the incident no longer produces
   `relational willingness must cite current episode evidence` when the
   current evidence row is the canonical `tool_result` `e1`.
2. `tool_result` maps to `current_episode` in model-facing provenance and is
   included in ordinary-goal current-episode handle feedback.
3. A relational decision citing only historical evidence remains rejected, and
   invalid relationship-state/stance pairings remain rejected.
4. The accepted-task result source remains `trigger_source=tool_result` with
   percept `source_kind=tool_result`; no persisted or public schema changes
   occur.
5. The individually inspected real-LLM case produces a valid, grounded
   clarification-oriented ordinary goal that cites the current tool result,
   without raw identifiers or invented facts.
6. Existing patched result-delivery tests continue to show non-empty dialog
   reaches `handle_send_message`, and focused background/accepted-task tests
   pass.
7. No code adds a deterministic dialog fallback, alters delivery retry policy,
   or changes task-resolution semantics.
8. The plan records test results, live review evidence, parent diff review,
   residual risk, and final workspace status before lifecycle closeout.

## Recorded Verification Evidence

- Pre-handoff baseline: the five DeepSeek-owned files were clean relative to
  `HEAD`; only the plan and registry row were changed at handoff.
- Focused deterministic verification: `90 passed` across the accepted-task
  prompt, relational-willingness, dependency, Cognition Core V2 contract, and
  background-delivery suites.
- Adjacent deterministic verification: `80 passed` across accepted-task
  lifecycle, background-job, task-resolution contract/resume/inline/
  orchestrator/specialist/state suites.
- Individual live verification: the explicit `-m live_llm` case passed in
  `9.93 seconds`; the model route was
  `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` using
  `gemma-4-31b-fable-5-agent-distill`. The inspected bid cited `e1`, retained
  `tool_result`, rendered `current_episode`, and requested the missing input.
- Live evidence: raw prompt/output/parsed bid/route metadata are in
  `test_artifacts/diagnostics/background_tool_result_delivery_live__1785925866073844500.json`;
  the parent judgment is in
  `test_artifacts/diagnostics/background_tool_result_delivery_live_review.md`.
- Compile and whitespace verification: Cognition Core V2 `compileall` passed
  and `git diff --check` passed.

## Parent Review And Closeout

The parent reviewed the complete diff and confirmed that the implementation
uses one canonical `episode`/`tool_result` current-episode source set, keeps
the strict validator and source schema intact, and leaves result construction,
retry policy, service ordering, dialog fallback behavior, and dispatcher
ownership unchanged. The deterministic history-only rejection and invalid
pairing coverage remain passing.

The live case uses the tracked `personalities/example.json` fixture because
the workspace does not contain the pre-existing live fixture's Asuna identity
file. It validates the changed cognition boundary rather than claiming a live
QQ send or a service trace id; the existing deterministic dispatcher test
continues to cover the send boundary. This limitation is recorded in the
parent live-review artifact.

Final workspace review found only the scoped production/test changes, the
completed plan registry/archive update, and ignored diagnostic artifacts.

## Progress Checklist

- [x] Production task, accepted-task row, conversation history, event log, and
  protected failure capsules inspected.
- [x] Canonical `tool_result` source construction and downstream evidence
  authority mismatch confirmed.
- [x] Scope fixed to authority projection and focused regression coverage.
- [x] User approval recorded.
- [x] Baseline reproduction recorded.
- [x] Implementation completed within the listed change surface.
- [x] Deterministic verification completed.
- [x] Individual live-LLM result-delivery case inspected.
- [x] Parent review and lifecycle closeout completed.
