# background task result blocker detail delivery bugfix

## Summary

- Goal: deliver the exact typed blocker and remaining limitation when a
  background task reaches a non-success task-resolution state, so cognition
  receives an actionable error instead of a generic clarification sentence.
- Status: completed
- Scope boundary: deterministic projection from a validated
  TaskResolutionResultV1 through the background worker into accepted-task and
  tool_result delivery text.
- Change direction: enrich the existing prompt-safe summary with the existing
  prompt-safe coding-run summary and remaining_needs; preserve all task states,
  schemas, retry behavior, and dialog ownership.
- Acceptance state: completed; all acceptance criteria are evidenced within the
  fixed delivery-only scope.

## Confirmed Decisions

- The delivery layer will make the error explicit from fields already present
  in the validated task-resolution result.
- The canonical detail order is:

  1. the existing prompt_safe_summary;
  2. Specific blocker: <coding_run_context.summary> when a non-empty,
     non-duplicate coding-run summary exists;
  3. Remaining limitation: <remaining_needs> when one or more remaining needs
     exist.

- The incident-shaped result must produce delivery text containing both:

  ~~~text
  Specific blocker: Please narrow the question before more source reading.
  Remaining limitation: Source-reading report limit would be exceeded.
  ~~~

- The existing needs_user_input result kind remains a failed accepted-task
  completion with its existing remaining_needs and sanitized coding-run
  context.
- The normal cognition and dialog path remains responsible for the final
  visible wording. The delivery projection supplies explicit evidence; it does
  not generate a deterministic fallback dialog or dictate character stance.
- The implementation owner is the delegated DeepSeek agent. The parent agent
  owns independent review, any in-scope remediation, verification sign-off,
  and lifecycle closure.

## Scope And Change Direction

The incident already contains the required detail in the task-resolution
result:

~~~text
prompt_safe_summary:
  The task needs additional user-provided information.
coding_run_context.summary:
  Please narrow the question before more source reading.
remaining_needs:
  Source-reading report limit would be exceeded.
~~~

The current background worker returns only prompt_safe_summary for
needs_user_input, approval_required, unavailable, and failed results. The
accepted-task failure record and background-job result therefore lose the
specific blocker before result_source.py constructs the canonical tool_result
episode.

The target flow is:

~~~text
validated TaskResolutionResultV1
  -> background worker composes explicit prompt-safe detail
  -> accepted-task failure_summary and job result_summary contain the detail
  -> result_source.py consumes the enriched result_summary
  -> cognition receives a concrete blocker and remaining limitation
  -> dialog asks for the concrete next input or scope clarification
~~~

This plan fixes the delivery projection only. It does not alter the upstream
coding-agent source selection, repository routing, source-reading cap, or
blocked-run generation.

## Mandatory Skills

- development-plan: governs this plan's lifecycle, implementation boundary,
  review, acceptance, and closeout.
- local-llm-architecture: preserves deterministic delivery ownership and LLM
  ownership of the final clarification wording.
- py-style: applies to the Python worker and test changes.
- test-style-and-execution: applies to focused deterministic tests and any
  individually inspected live-LLM delivery case.
- debug-llm: applies when validating that the enriched result produces a
  grounded clarification rather than a generic request.
- python-venv: applies to every Python and pytest command.
- cjk-safety: applies if Python tests or fixtures add or refactor CJK strings.

## Mandatory Rules

- Use venv\Scripts\python.exe for Python checks and tests.
- Keep the source kind tool_result, accepted-task states, task-resolution
  schemas, coding-run context schema, delivery retry policy, and dispatcher
  contract unchanged.
- Read only the validated prompt-safe fields: prompt_safe_summary,
  coding_run_context.summary, and remaining_needs.
- Preserve exact field ownership. The worker may compose transport text but
  must not infer a new semantic blocker, convert a status, or rewrite an LLM
  stance.
- Deduplicate identical detail lines and preserve deterministic ordering.
- Do not expose coding_run_ref, raw worker payloads, repository paths,
  internal action names, credentials, or adapter metadata in the composed
  delivery text.
- Preserve the existing bounded artifact behavior based on the job's declared
  max_output_chars.
- Run deterministic tests in batches. Run each live-LLM case individually,
  inspect its output, and record the quality judgment.
- Preserve unrelated worktree changes and keep the production database
  read-only during verification.

## Must Do

1. Update the background worker's non-success delivery projection so the
   existing coding-run blocker summary and remaining limitations reach the
   accepted-task failure summary and background-job result summary.
2. Keep resolved and evidence-bearing partial result projection unchanged,
   including evidence ordering, source URL projection, and partial limitation
   handling.
3. Keep the existing accepted-task failure transition unchanged apart from
   receiving the enriched summary. needs_user_input remains
   completion_status=failed with the same result_kind.
4. Add deterministic regression coverage for:
   - a coding-run needs_user_input result containing the incident-shaped blocker
     and limitation;
   - a non-coding needs_user_input result containing only remaining_needs;
   - missing blocker details, which must preserve the existing generic summary
     without inventing text;
   - duplicate blocker/limitation text, which must appear once;
   - _complete_task_orchestrator_job, proving the enriched text reaches both
     accepted-task failure state and completed background-job result state;
   - build_result_ready_episode_from_job, proving the enriched result_summary
     becomes the semantic summary of the canonical tool_result episode.
5. Update the background-work ICD to state that non-success task-resolution
   delivery preserves the typed blocker and remaining limitation in the
   prompt-safe result summary.
6. Run one individually inspected result-delivery LLM case using the
   incident-shaped result. The review must verify that the model receives the
   concrete blocker and produces a clarification directed at narrowing the
   question or supplying the named remaining input.
7. Record implementation evidence, focused test results, live review
   evidence, parent diff review, and final workspace status before closing the
   plan.

## Deferred

- All changes under src/kazusa_ai_chatbot/coding_agent/**.
- Changes to src/kazusa_ai_chatbot/task_resolution/specialists/coding.py,
  task_resolution/orchestrator.py, task-resolution status mapping, source URL
  propagation, source selection precedence, repository routing, source
  reading, report limits, or coding-run evidence collection.
- Converting a blocked coding run into partial, replaying its evidence, or
  delivering source analysis that the coding agent did not complete.
- Changes to src/kazusa_ai_chatbot/background_work/result_source.py; it will
  consume the enriched result_summary through its existing contract.
- Changes to accepted_task schemas, lifecycle transitions, delivery retry
  counts, lease recovery, dispatcher behavior, adapters, or conversation
  persistence.
- Deterministic fallback dialog generation after cognition failure.
- Prompt rewrites, new LLM stages, new result statuses, compatibility aliases,
  migrations, production requeue, and unrelated cleanup.

## Target State

For a validated non-success task result with the incident-shaped fields, the
worker's delivery summary is:

~~~text
The task needs additional user-provided information.
Specific blocker: Please narrow the question before more source reading.
Remaining limitation: Source-reading report limit would be exceeded.
~~~

The following state remains unchanged:

- accepted-task state=failure_ready before normal result delivery;
- accepted-task completion_status=failed;
- accepted-task result_kind=needs_user_input;
- accepted-task remaining_needs;
- accepted-task coding_run_context;
- background-job task_resolution_result;
- background-job delivery state transitions;
- result episode trigger_source=tool_result;
- percept source_kind=tool_result;
- cognition and dialog ownership of the visible response.

When no structured blocker or remaining limitation exists, the worker preserves
the existing prompt_safe_summary and emits no invented explanation.

## Contracts And Data Shapes

The input contract remains task_resolution_result.v1:

~~~json
{
  "status": "needs_user_input",
  "prompt_safe_summary": "The task needs additional user-provided information.",
  "remaining_needs": [
    "Source-reading report limit would be exceeded."
  ],
  "coding_run_context": {
    "schema_version": "coding_run_context.v1",
    "status": "blocked",
    "summary": "Please narrow the question before more source reading.",
    "limitations": [
      "Source-reading report limit would be exceeded."
    ],
    "allowed_next_actions": [
      "respond_to_blocker",
      "summarize",
      "status",
      "cancel"
    ],
    "followup_open": true
  }
}
~~~

The delivery text contract is deterministic:

~~~text
<prompt_safe_summary>
Specific blocker: <coding_run_context.summary>
Remaining limitation: <remaining_needs joined with "; ">
~~~

The worker omits an empty field and omits exact duplicate lines. It does not
add a new persisted field or mutate the validated result object. Existing
resolved and partial evidence projection remains the source of their delivery
text.

## Change Surface

### Delete

- None.

### Modify

- src/kazusa_ai_chatbot/background_work/worker.py: update
  _task_result_delivery_summary or its local deterministic helper to preserve
  explicit non-success blocker detail while retaining the current
  resolved/partial projection.
- src/kazusa_ai_chatbot/background_work/README.md: document the non-success
  result-summary projection and its prompt-safe field ownership.
- tests/test_task_resolution_background_resume.py: add worker summary and
  accepted-task/job propagation coverage.
- tests/test_background_work_delivery.py: assert the enriched result summary
  is the semantic text of the canonical tool-result episode.

### Create

- test_artifacts/diagnostics/background_task_result_blocker_detail_delivery_live_review.md:
  parent-authored review of the individually inspected live result-delivery
  case.
- test_artifacts/diagnostics/background_task_result_blocker_detail_delivery_live_*.json:
  raw or protected review artifacts for that case, using the existing ignored
  diagnostic-artifact convention.

### Keep

- src/kazusa_ai_chatbot/background_work/result_source.py: existing
  result_summary-first semantic projection and canonical tool_result source.
- src/kazusa_ai_chatbot/accepted_task/**: existing lifecycle, state, and
  sanitized context ownership.
- src/kazusa_ai_chatbot/background_work/delivery.py: existing cognition,
  dispatcher, retry, and delivery-finalization ownership.
- src/kazusa_ai_chatbot/cognition_core_v2/**: existing evidence authority,
  goal, dialog, and surface contracts.
- src/kazusa_ai_chatbot/coding_agent/**: upstream coding behavior and blocker
  generation remain outside this plan.

## Agent Autonomy Boundaries

The implementation agent may choose the local helper name, exact test helper
placement, assertion decomposition, and Markdown wording within the fixed
delivery text labels and file surface.

The implementation agent must not change any upstream coding-agent or
task-resolution semantics, add a new result status, preserve unvalidated raw
evidence, expose internal coding-run identifiers, add a fallback sender,
rewrite prompts, alter retry policy, or broaden the result-source schema.

If implementation requires a change outside the listed delivery surface, the
agent records the conflict and pauses for a plan amendment. It does not
silently expand the plan.

## Verification

1. Capture the worktree baseline and run the focused pre-change tests for
   background resume and result delivery.
2. Run the deterministic suites:

   ~~~powershell
   venv\Scripts\python.exe -m pytest tests/test_task_resolution_background_resume.py tests/test_background_work_delivery.py tests/test_background_work_jobs.py tests/test_accepted_task_lifecycle.py -q
   ~~~

3. Verify the incident-shaped summary directly through the worker helper and
   through _complete_task_orchestrator_job, including accepted-task and
   job-result arguments.
4. Verify the result-source episode retains trigger_source=tool_result,
   percept source_kind=tool_result, and the complete explicit blocker text.
5. Run the individually selected live-LLM delivery case with -q -s, inspect
   the rendered input, model output, parsed contract, trace status, and human
   quality judgment, then save the required review artifact.
6. Run the adjacent task-resolution and background-work deterministic suites
   without modifying their production semantics.
7. Run:

   ~~~powershell
   venv\Scripts\python.exe -m compileall -q src/kazusa_ai_chatbot/background_work
   git diff --check
   ~~~

8. Review the complete diff and confirm that no path under
   src/kazusa_ai_chatbot/coding_agent/** or the deferred task-resolution
   surfaces changed.

## Acceptance Criteria

1. The incident-shaped needs_user_input result delivers the exact Specific
   blocker and Remaining limitation lines to both the accepted-task failure
   summary and background-job result summary.
2. The canonical result-source episode exposes the enriched text as its
   semantic summary while retaining tool_result source identity and the
   existing delivery path.
3. Non-coding user-input, approval, unavailable, and failed results preserve
   their prompt-safe summary and append only their validated remaining needs.
4. Empty or duplicate detail fields produce no invented or duplicated
   explanation.
5. Resolved and partial evidence delivery remains behaviorally unchanged.
6. The individually inspected live-LLM case receives the concrete blocker and
   produces a grounded clarification that identifies the needed narrowing or
   input instead of only asking for unspecified materials.
7. Focused and adjacent deterministic tests, compile checks, and whitespace
   checks pass.
8. The final diff contains only the approved delivery worker, documentation,
   test, diagnostic-review, plan, and registry surfaces; no coding-agent
   implementation changes are present.

## Progress Checklist

- [x] User approves this draft for execution.
- [x] DeepSeek receives the approved plan with the delivery-only file scope.
- [x] Pre-change baseline and focused tests are recorded.
- [x] Worker projection and deterministic tests are implemented.
- [x] Background ICD wording is updated.
- [x] Individually inspected live-LLM delivery evidence is recorded.
- [x] Parent review confirms explicit blocker propagation and scope fidelity.
- [x] Parent remediation and final verification are complete.
- [x] Plan and registry lifecycle are closed with acceptance evidence.

## Execution Evidence

- Baseline: the pre-change worktree contained only this user-provided plan as
  an untracked change; the focused background-resume and result-delivery
  suites passed with 29 tests.
- Implementation: DeepSeek modified only `worker.py`, the background-work
  ICD, and the two focused test files. The parent corrected one invalid test
  checkpoint fixture and changed the validated coding-run summary read to
  plain indexing during review.
- Deterministic verification: the post-change focused suites passed with 36
  tests; `test_background_work_jobs.py` and
  `test_accepted_task_lifecycle.py` passed with 26 tests. The background-work
  package compiled with `compileall`, and `git diff --check` passed.
- Live evidence: one individually inspected incident-shaped real-LLM case
  used the `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` route. Both exact detail
  lines reached the model-facing `tool_result` evidence; the final parsed goal
  asked for the narrowing direction and the input needed to avoid the
  source-reading report limit. The human review is
  `test_artifacts/diagnostics/background_task_result_blocker_detail_delivery_live_review.md`
  and the raw trace is
  `test_artifacts/diagnostics/background_task_result_blocker_detail_delivery_live__1786086213122300700.json`.
- Independent review: a separate DeepSeek review found no blocking defects;
  its only low-severity observation was remediated by the parent. It confirmed
  the resolved/partial branch, result-source identity, prompt-safe ownership,
  and deferred-scope boundaries.
- Scope audit: no file under `src/kazusa_ai_chatbot/coding_agent/**`,
  `src/kazusa_ai_chatbot/task_resolution/**`, or
  `src/kazusa_ai_chatbot/background_work/result_source.py` changed.
- Final workspace status at closeout: modified
  `development_plans/README.md`,
  `src/kazusa_ai_chatbot/background_work/README.md`,
  `src/kazusa_ai_chatbot/background_work/worker.py`,
  `tests/test_background_work_delivery.py`, and
  `tests/test_task_resolution_background_resume.py`; the completed plan is an
  untracked archived record at this path. Diagnostic JSON and Markdown review
  artifacts remain under the repository's ignored `test_artifacts` paths.

## Execution Handoff

When approved, the DeepSeek implementation handoff owns only the listed
background worker, background-work documentation, and focused test files. The
handoff must state the baseline, exact output contract, deferred coding-agent
scope, applicable skills, and the next verification checkpoint. The parent
agent reviews the complete diff, fixes findings within the same surface,
records live and deterministic evidence, and closes the plan only after all
acceptance criteria are evidenced.
