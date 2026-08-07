# cognition core v2 action-planning request-fidelity bugfix plan

## Summary

- Goal: preserve the user's requested retrieval effect when Cognition Core V2
  converts an admitted goal into a resolver request.
- Status: completed
- Scope boundary: action-planning prompt/input projection, its semantic
  contract tests, and the downstream task-resolution routing regression.
- Change direction: make the action planner distinguish the requested effect
  from capability, permission, and feasibility metadata, while retaining the
  existing resolver request shape and deterministic ownership boundaries.
- Acceptance state: the captured full-input replay must produce a retrieval
  objective and the downstream orchestrator must receive that objective rather
  than a capability-audit objective.
- Implementation authority: production implementation begins only after this
  draft is approved.

## Confirmed Decisions

- The first fix targets action planning and its task-resolution handoff.
- Coding-agent behavior and dialog generation are outside this bugfix.
- RAG2/conversation evidence and RAG3/local-context implementations remain
  unchanged; the bugfix must make their existing capability reachable with the
  correct semantic objective.
- The protected trace and background-job export already captured for cognition
  run `63f34d5ae7b342a7af1e77afa05d45b4` are the reproduction source of truth.
- The full captured input is required for regression coverage. The smaller
  one-evidence-row fixture remains a negative control because it does not
  reproduce the production semantic drift.

## Original Failure Mode

### User request and captured run

The user asked:

```text
@一之濑明日奈 明日奈～能抓取一下 @Nagasaki-soyo-清尘 最近10天的聊天记录么？
```

The delivery tracking id was
`63f34d5ae7b342a7af1e77afa05d45b4`. The protected Cognition Core trace was
`llmtrace_e86e2bda365e49aca2d0ad54fb0fd066`.

The action-planning input used the production-sized captured payload:

- prompt length: 12,719 characters;
- 12 evidence rows, including the current request, recent conversation, and
  promoted context;
- the repaired ordinary-response bid from the failure capsule;
- the production action and resolver affordance roster;
- empty initial resolver goal progress;
- the captured resolver context.

The raw failure capsule and the full background execution context are retained
at:

- `test_artifacts/diagnostics/llm_trace_llmtrace_e86e2bda365e49aca2d0ad54fb0fd066_20260806T222404Z.json`
- `test_artifacts/diagnostics/background_job_63f34d5a.json`

### Action-planning failure

The user requested retrieval of a bounded conversation window. The action
planner instead emitted a `task_resolution_request` whose semantic goal was
equivalent to:

```text
核实当前角色是否具备抓取特定用户（@Nagasaki-soyo-清尘）最近 10 天聊天记录的技术能力及权限，并获取执行该操作所需的具体证据或限制条件。
```

The corresponding reason also framed the task as a background verification of
runtime support and permissions. This changed the task being delegated. The
planner preserved the target and time range, but replaced the requested
effect—retrieve the messages—with a meta-task—audit whether retrieval is
possible.

The existing action-planning contract says that the planner owns runtime
feasibility and resolver selection, but it does not authorize the planner to
replace the user's requested effect with a capability self-audit. Capability
availability and permission remain deterministic runtime or specialist
boundaries; they are not the semantic objective sent to a retrieval owner.

### Downstream manifestation

The task-resolution orchestrator then received the capability-audit objective
and made two live selections:

1. `local_context` for a subgoal about verifying role permissions and
   technical capabilities;
2. `coding` with `read_only` for analyzing system capabilities and API
   permissions.

The session ended `deferred` with no evidence. This route is consistent with
the wrong objective. It does not prove that the conversation retrieval owner
lacks capability, because the original retrieval objective never reached that
owner.

The original action-planning failure remains preserved in the baseline artifact
`test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__captured_full_action_planning_capability_audit.json`.
The post-fix full-input gate is
`tests/test_cognition_core_v2_action_planning_live_llm.py::test_captured_full_action_planning_preserves_retrieval_goal`.
The downstream failure-path replay is
`tests/test_task_resolution_live_llm.py::test_live_captured_chat_history_audit_routes_to_coding_without_evidence`;
its first local-context selection is pinned to the captured production decision,
while the subsequent coding selection remains live. The corrected downstream
gate is
`tests/test_task_resolution_live_llm.py::test_live_captured_chat_history_retrieval_routes_to_local_context`.

The human-readable review is
`test_artifacts/llm_traces/captured_chat_history_action_task_resolution_review.md`.

## Scope And Change Direction

The action planner will remain the single LLM owner of semantic capability
request selection. The change adds a request-fidelity boundary to its existing
prompt-facing input and output contract:

1. The planner receives evidence with a deterministic semantic authority label
   derived from the existing validated source-kind and memory-scope metadata.
   The current user episode is explicitly distinguishable from historical or
   contextual evidence.
2. The planner is instructed to identify the user's requested effect and its
   explicit target, scope, and time constraints before deciding answerability
   or resolver need.
3. A resolver request's `semantic_goal` remains the concrete effect that the
   user asked to obtain or perform. Missing evidence may be expressed as a
   dependency of that goal, but it must not replace the goal with a question
   about the system's capability, permission, feasibility, or API support.
4. A capability or permission audit remains valid only when the current user
   request explicitly asks whether the operation is possible or authorized.
5. `goal_resolution=requires_required_evidence` continues to mean that the
   requested answer requires evidence. It does not authorize a semantic rewrite
   of the resolver objective.
6. An empty `current_resolver_goal_progress` shell cannot receive an invented
   checklist. A contract violation enters the existing bounded same-stage
   regeneration path; no semantic keyword filter or post-LLM rewrite is used.
7. The resulting resolver request is passed unchanged through authorization
   and task resolution. Downstream stages validate and execute the objective;
   they do not repair an action-planning semantic substitution.

This is a big-bang model-facing contract correction. The existing output field
names, resolver handle vocabulary, request limits, authorization boundary, and
task-resolution public IO remain canonical. No alias, compatibility mapper,
keyword router, semantic post-filter, or deterministic replacement sentence is
introduced.

## Mandatory Skills

- `development-plan`: governs this draft, approval boundary, implementation
  scope, verification, and closeout.
- `local-llm-architecture`: governs planner/specialist ownership, prompt input
  shaping, bounded local-model behavior, and blast-radius control.
- `no-prepost-user-input`: prohibits keyword routing or post-LLM semantic
  rewriting of the user's request.
- `debug-llm`: requires raw live evidence and an agent-authored human review
  artifact for each live comparison or regression run.
- `test-style-and-execution`: governs deterministic contract tests and
  one-at-a-time inspected live LLM tests.
- `py-style`: applies to all Python implementation and test changes.
- `cjk-safety`: applies when editing the existing CJK action-planning prompt or
  CJK replay fixtures in Python files.
- `llm-trace-debug`: governs use of the protected trace and captured failure
  capsule as replay evidence.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python checks and tests.
- Inspect each live LLM artifact before running the next live case.
- Preserve raw prompt messages, raw model output, parsed output, selected route,
  and the downstream task-resolution decision in durable JSON artifacts.
- Author the Markdown quality review from the inspected raw artifacts. Test or
  script code may write raw evidence but may not generate the human judgment
  review.
- Keep deterministic code responsible for structural validation, provenance
  projection, capability availability, permissions, limits, and execution.
- Keep the LLM responsible for interpreting the current request, selecting
  whether required evidence is needed, and writing the semantic objective.
- Do not add a local keyword allowlist or disallowlist for terms such as
  capability, permission, feasibility, retrieval, or chat history.
- Do not change RAG, coding-agent, dialog, persistence, adapter, or delivery
  production code under this plan.
- Preserve all unrelated worktree changes and do not read `.env`.

## Contracts And Data Shapes

### Prompt-facing evidence projection

The existing action-planning human payload keeps the current evidence handles
and semantic text, and adds one deterministic field to each projected row:

```json
{
  "handle": "e1",
  "source_kind": "episode",
  "provenance_role": "current_episode",
  "semantic_text": "the current user request"
}
```

`provenance_role` is derived only by the existing
`project_evidence_provenance_role(...)` contract. For a user-message episode,
`current_episode` is the authoritative current request and scene. Historical
conversation, memory, reflection, and contextual rows remain supporting
evidence with their existing source ownership. The model receives no raw
platform id, database id, adapter syntax, or storage metadata.

### Resolver semantic-goal contract

The existing resolver request shape remains:

```json
{
  "bid_handle": "b1",
  "resolver_handle": "r4",
  "semantic_goal": "retrieve @Nagasaki-soyo-清尘's chat history from the last 10 days",
  "reason": "the answer requires the requested conversation evidence"
}
```

The `semantic_goal` must:

- preserve the requested effect;
- preserve user-supplied target names, scope, and temporal constraints;
- state the evidence or bounded work needed to advance that effect;
- remain free of specialist names, queue/timing settings, backend parameters,
  permission decisions, and final wording.

The `reason` explains why the request advances the admitted bid. It may state
that evidence is missing, but it is not a second task objective.

An explicit user capability question keeps its meaning. For example, a user
asking whether the character is authorized to read another user's history may
produce a capability-audit semantic goal. A direct request to retrieve the
history must produce the retrieval semantic goal even though deterministic
runtime checks will still enforce permissions before execution.

When the current resolver-progress object is an empty shell, the action planner
returns `resolver_goal_progress=null`. An attempted new checklist is a
structural contract error and receives one of the existing bounded same-stage
replacement attempts; an already populated checklist may receive only a local
update that remains faithful to the current requested effect.

## Must Do

1. Update `ACTION_PLANNING_PROMPT` with a short generation procedure that
   distinguishes current-request effect, admitted character motive, supporting
   evidence, and runtime capability metadata.
2. State in that prompt that `semantic_goal` is a faithful continuation of the
   requested effect. State that capability, permission, feasibility, and API
   support are execution constraints unless the user explicitly asks for an
   audit of them.
3. Project `provenance_role` into action-planning evidence rows using the
   existing deterministic provenance helper, and keep the projection bounded
   under the current prompt cap.
4. Preserve the existing `goal_resolution` semantics and resolver request
   schema. Do not add a second objective field or a compatibility vocabulary.
5. Add deterministic prompt-contract coverage for current-request authority,
   effect preservation, explicit capability-question handling, and the absence
   of deterministic semantic rewriting.
6. Convert the captured action-planning live test from a failure detector into a
   post-fix regression gate. It must replay the full protected input, retain the
   target and ten-day constraint, and manually review whether the produced
   objective is retrieval rather than capability auditing.
7. Add a paired live boundary case where the user explicitly asks about
   capability or permission. The planner must retain that audit meaning, proving
   the fix does not force every data request into retrieval.
8. Add a downstream live task-resolution guard using the corrected retrieval
   objective and the captured execution context. The orchestrator must select
   the local-context/conversation evidence path and must not select coding for
   this objective. Stop the guard before any coding handler is entered.
9. Update the Cognition Core V2 README and the inspected Markdown review to
   document the corrected semantic ownership and the evidence boundary.
10. Keep the existing full-input raw artifacts and add the post-fix raw output,
    parsed output, route summary, downstream selection, and review notes beside
    them.

## Deferred

- Changes to RAG2 conversation retrieval, RAG3 local-context retrieval, or any
  source-owned retrieval worker.
- Changes to task-resolution specialist prompts, coding selection policy, task
  budgets, checkpoint state, background queueing, or worker persistence.
- Changes to coding-agent implementation, coding-agent dialog, dialog
  generation, final wording, adapter delivery, or delivery receipts.
- New resolver capabilities, action capabilities, specialist handles, output
  fields, retries, repair models, or prompt-budget increases.
- Deterministic semantic classifiers, keyword filters, lexical rejection of
  model output, or post-LLM rewriting of `semantic_goal`.
- Production database writes, deployment, rollout flags, or compatibility
  support for the incorrect capability-audit objective.

## Target State

```text
current user request
  -> goal cognition emits an admitted character motive
  -> action planner preserves the requested effect and decides evidence need
  -> resolver authorization checks evidence need and capability match
  -> task-resolution orchestrator receives the retrieval objective
  -> existing local-context/conversation evidence owner retrieves evidence
  -> later cognition/dialog stages explain the result
```

For the captured request, the action-planning result has:

- `goal_resolution=requires_required_evidence`;
- one `task_resolution_request` when authorization accepts it;
- a semantic goal that asks for the target member's last ten days of chat
  history;
- no capability-audit substitution;
- no coding request or coding specialist selection.

For an explicit capability question, the same contract permits a capability
audit objective. The distinction is model-owned semantic interpretation based
on the current request and typed evidence, not a deterministic string rule.

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`: project
  provenance roles into the action-planning payload, revise the static
  semantic-goal generation contract, and validate empty progress shells.
- `tests/test_action_selection_prompt_contract.py`: assert the new request
  fidelity and capability-question boundaries.
- `tests/test_action_selection_payload.py`: cover the model-facing ownership
  wording and payload contract.
- `tests/test_cognition_core_v2_action_planning_bugfix.py`: add deterministic
  projection and semantic-contract coverage without mocking a live quality
  result.
- `tests/test_cognition_core_v2_action_planning_live_llm.py`: replay the exact
  captured failure input and add the explicit capability-question boundary
  case, one test at a time.
- `tests/test_task_resolution_live_llm.py`: add the corrected-objective
  downstream routing guard while leaving coding-agent execution unentered.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`: document that action
  planning preserves user-requested effect while runtime owners enforce
  capability and permission.
- `test_artifacts/llm_traces/captured_chat_history_action_task_resolution_review.md`:
  extend the agent-authored review with before/after semantic objectives and
  the downstream route result.
- `development_plans/README.md`: register this active bugfix plan as in
  progress.

### Create

- `test_artifacts/llm_traces/cognition_core_v2_action_planning_live_llm__*.json`:
  raw and parsed live evidence following the existing ignored-artifact
  convention.
- `test_artifacts/llm_traces/cognition_core_v2_action_planning_request_fidelity_review.md`:
  parent-authored review of the original replay, explicit capability boundary,
  and corrected replay.

### Keep

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: existing evidence,
  provenance, goal-resolution, and resolver-request contracts remain
  authoritative.
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`:
  authorization continues to check unresolved evidence and capability match;
  it does not rewrite semantic goals.
- `src/kazusa_ai_chatbot/task_resolution/`: existing specialist roster,
  orchestrator limits, checkpoint, and public IO.
- `src/kazusa_ai_chatbot/rag/`: existing conversation and local-context
  evidence ownership.
- Coding-agent and dialog packages: no changes under this plan.

## Agent Autonomy Boundaries

The implementation agent may choose local helper names, exact prompt paragraph
ordering, deterministic provenance-projection decomposition, test fixture
construction, and Markdown wording while preserving the contracts and file
surface above.

The implementation agent must not add a semantic post-filter, keyword route,
fallback objective, alternate output shape, hidden capability audit, or change
to the RAG, task-resolution, coding, or dialog ownership boundaries. It must
not make the explicit-capability boundary pass by hard-coding the captured
target, user name, date range, or Chinese terms into production logic or the
reusable prompt as a test-shaped example.

If the stated request-fidelity contract cannot be implemented within the
listed action-planning surface, implementation pauses for a plan amendment.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Action-planning prompt | bigbang | Replace the semantic instruction in place and update its contract tests together. |
| Prompt-facing evidence projection | bigbang | Add the typed provenance label to the canonical payload; do not preserve a parallel unlabelled payload. |
| Resolver request output | compatible | Keep the existing fields, handles, limits, and authorization IO unchanged. |
| Downstream task resolution | compatible | Consume the corrected semantic objective through the existing public request. |
| Tests and review evidence | bigbang | Replace the old failure expectation with the post-fix regression and retain the original raw failure as baseline evidence. |

## Verification

1. Capture the existing worktree baseline and run focused deterministic action
   planning contract tests before implementation.
2. Run the captured full action-planning replay individually with
   `-m live_llm -q -s`; inspect its raw JSON and parsed result before running
   the next live case.
3. Run the explicit capability/permission-question boundary individually and
   inspect its raw output and review notes.
4. Run the corrected-objective task-resolution routing guard individually with
   the production-sized captured execution context; verify the selected
   specialist and stop before coding execution.
5. Run the focused deterministic action-planning, prompt-contract, and
   task-resolution contract suites with `venv\Scripts\python.exe`.
6. Verify that the corrected prompt payload contains typed provenance roles,
   remains under the existing action-planning prompt cap, and contains no raw
   identifiers or adapter syntax.
7. Verify that the full replay's empty progress shell is rejected and repaired
   to `resolver_goal_progress=null` without changing the retrieval request.
8. Run `py_compile` for every changed Python file, `git diff --check`, and a
   final complete diff/scope review.
9. Review the agent-authored Markdown evidence against the raw input/output;
   record the before/after semantic objective, route, downstream selection, and
   residual risk.

## Acceptance Criteria

1. The exact full-input replay of the captured run no longer changes a direct
   chat-history retrieval request into a capability/permission audit.
2. The corrected resolver semantic goal preserves the requested target and
   ten-day temporal scope and is suitable for the existing conversation
   evidence owner.
3. `goal_resolution=requires_required_evidence` remains valid when evidence is
   needed, without changing the semantic objective into a feasibility query.
4. An explicit user capability or permission question remains an audit goal,
   proving that the fix preserves semantic distinctions instead of forcing a
   universal retrieval route.
5. The downstream task-resolution orchestrator receives the corrected
   retrieval objective and selects the local-context/conversation evidence path
   rather than coding; no coding handler is entered.
6. Existing resolver request shape, authorization, task-resolution public IO,
   RAG ownership, action limits, and failure-closed structural validation remain
   unchanged.
7. Deterministic tests, individually inspected live tests, raw evidence,
   agent-authored review, compile checks, whitespace checks, documentation, and
   final scope review all pass.

## Progress Checklist

- [x] User approves this draft for implementation.
- [x] Pre-change focused deterministic checks and known reproductions are
      recorded.
- [x] Action-planning provenance projection and request-fidelity prompt are
      implemented.
- [x] Deterministic prompt and payload contract tests pass.
- [x] Captured full-input replay passes the post-fix semantic rubric.
- [x] Explicit capability-question boundary passes.
- [x] Corrected objective reaches the local-context/conversation route without
      entering coding.
- [x] Raw evidence and parent-authored review are inspected and recorded.
- [x] README, plan registry, and final scope review are complete.
- [x] Plan lifecycle is closed with acceptance evidence.

## Acceptance Evidence

- The baseline full-input artifact preserves the original capability-audit
  substitution; the final full replay preserves the target member and ten-day
  retrieval objective with `requires_required_evidence` and null goal progress.
- The explicit capability-question replay retains an audit objective, while the
  corrected task-resolution replay selects `local_context`, returns validated
  evidence, and never enters coding.
- The negative-control live replay was rerun under the final empty-shell guard;
  its final artifact records the bounded repair and `resolver_goal_progress` as
  `null`.
- Focused deterministic action-planning and task-resolution suites pass 62
  tests; all six changed Python files compile; `git diff --check` is clean.
- The parent-authored review and final DeepSeek read-only review found no
  blocker, high, or medium issue. The final reviewer verdict is `approve`.
