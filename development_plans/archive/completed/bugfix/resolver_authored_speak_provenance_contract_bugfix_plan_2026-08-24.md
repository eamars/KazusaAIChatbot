# Resolver-Authored Speak Provenance Contract Bugfix Plan

## Summary

- Goal: make every resolver-authored user-visible `speak` action satisfy the
  caller-owned target-role provenance required by text L3.
- Status: completed
- Date: 2026-08-24
- Approval: the user explicitly approved a separate plan and implementation,
  conditional on a complete justification and system-level analysis before
  any previously unplanned production edit.
- Scope boundary: the deterministic cognition-resolver fallback-action owner;
  no prompt, model, memory, adapter, database, emotion, dialog, or L3 behavior
  change.
- Change direction: big-bang producer-contract repair across all three
  resolver-authored visible fallback builders.
- Acceptance state: passed deterministic RED/GREEN verification, independent
  parent review, and a user-approved branch-targeted real-service gate; the
  semantic-progression gates may resume.

## Failure Evidence And Full Analysis

### Observed service failure

The first memory-enabled service turn for the active semantic-progression plan
used the disclosed input:

```text
我最近工作有点乱，你会怎么帮我排第一步？
```

The canonical brain `/chat` request reached the real cognition graph. P selected
`goal_resolution=requires_user_input` and requested `human_clarification`.
The resolver created a pending resume, reran cognition, detected the repeated
clarification request, cleared it, and constructed a visible fallback `speak`.
Text L3 then raised:

```text
ValueError: speak action cognition provenance is required
```

Evidence:

- trace: `llmtrace_6595ac642fe649bc9c065bfccc6b9c3e`;
- artifacts:
  `test_artifacts/debug_runs/semantic_response_progression_20260824_service/l2a/`;
- service log records `Resolver blocked capability repeated after pending
  resume creation` immediately before the invariant failure;
- the trace contains the initial and pending-resume A1/A2/G/P passes and no L3
  or dialog call;
- `final_dialog_count=0`, Conversation Progress remained empty, and the user
  memory lookup returned zero rows.

The real turn therefore establishes the operational failure but supplies no
semantic quality verdict for the active progression plan.

### Exact control path

```text
P requests human_clarification
  -> resolver capability returns blocked observation
  -> pending resume is persisted
  -> final cognition repeats the same resolver request
  -> loop clears repeated resolver request and cognition action specs
  -> _pending_resume_speak_action_spec(...) creates fallback speak
  -> generic validate_action_spec(...) accepts it
  -> persona route selects visible text
  -> L3 reads speak.cognition_provenance.target_roles
  -> field is absent; L3 fails before its model call
```

The same producer omission exists in all resolver-authored visible fallback
builders:

1. `_pending_resume_speak_action_spec(...)`;
2. `_user_input_blocker_speak_action_spec(...)`;
3. `_terminal_blocker_speak_action_spec(...)`.

The third builder serves duplicate-request, max-cycle, and mixed-lifecycle
terminal surfaces, so repairing only the observed pending-resume case would
leave the same failure mode active on other deterministic branches.

### Historical cause

The resolver fallback builders entered the codebase in commit `93d0ce13` on
2026-06-01. The caller-owned L3 addressee contract entered later in commit
`cd0fe945` on 2026-08-23. That later design intentionally removed an LLM
preference/copy stage and made deterministic code project exact addressee rows
from the selected `speak` action's `cognition_provenance.target_roles`.

The later contract updated canonical cognition action materialization and L3
fixtures, but it did not reconcile the older resolver-owned manual action
builders. Existing resolver tests check the fallback action kind, surface role,
continuation reference, and content requirements. They neither assert
`cognition_provenance` nor pass the fallback through the L3 addressee boundary.
The defect is therefore a producer/consumer contract split plus a missing
cross-boundary test.

### Relationship to the semantic-progression change

The active semantic-progression diff changes Conversation Progress projection,
model-facing progression context, cognition/L3 prompt payloads, and dialog
fidelity. It changes none of:

- `cognition_resolver/loop.py`;
- generic action-spec materialization;
- the persona graph route;
- the L3 provenance guard;
- adapter forwarding.

Its new `overused_moves` value was empty on this first turn. The operational
failure is a latent resolver/action integration defect exposed by the real
gate, not a regression caused by the progression implementation.

Promoted memory did influence P's semantic choice toward an exchange-condition
interaction. That semantic evidence remains relevant to the progression RCA,
but it did not create the missing Python field or the L3 exception. This plan
only restores executability so the original memory/personality behavior can be
judged through complete visible turns.

### Adapter and persistence boundaries

The debug adapter forwards `req.model_dump()` to the same canonical brain
`/chat` endpoint. The exception occurred after queue processing inside the
brain persona graph, so adapter behavior is not causal.

The pending-resume branch can write its normal pending ledger row before L3.
This plan adds no new database write, schema, collection, cleanup, migration,
or reset. The failed test identity will not be reused for acceptance because
its pending row can affect later cognition.

## System Contract Audit

### Pipeline roles

- P owns whether clarification or another resolver capability is semantically
  required.
- The cognition resolver owns bounded capability execution, recurrence caps,
  pending state, and deterministic terminal/fallback action construction.
- Action specs carry delivery and source lineage.
- Text L3 owns content planning and deterministic addressee projection from
  caller-owned target roles.
- Dialog owns wording within the accepted L3 plan.
- Adapters deliver the brain response and do not repair brain state.

### Smallest current contract

```text
Semantic question: none; this fix adds no LLM judgment.
Required inputs: the already resolved current global user id in persona state.
Required output: cognition_provenance with one current-user target role and an
empty evidence-handle list on each resolver-authored visible speak action.
Deterministic owner: cognition_resolver.loop.
Rejected complexity: prompts, retries, schema versions, compatibility paths,
new model calls, DB reads/writes, adapter logic, semantic classifiers, and
post-generation repair.
Evidence required: all three producer branches pass exact deterministic tests,
the L3 boundary accepts a produced action, and the captured service case reaches
L3/dialog without the invariant failure.
```

### Canonical target ownership

A resolver fallback surface exists only when
`_should_surface_terminal_blocker(...)` confirms a `user_message` cognitive
episode. Its semantic addressee is the current user who owns that message,
including in group chat. Delivery remains `current_channel`; the semantic role
must remain a direct `user` target rather than an empty list or group broadcast.

The resolver will derive the target from the already resolved required
`GlobalPersonaState.global_user_id`. A missing or blank current-user id is an
internal contract error and fails closed before L3. This does not expose the id
to a model: L3 converts the role to the prompt-safe `current_user` handle and
visible display name.

### Provenance shape

Every resolver-authored visible fallback action must carry:

```json
{
  "cognition_provenance": {
    "target_roles": [
      {
        "role": "target",
        "entity_kind": "user",
        "entity_id": "<resolved current global user id>"
      }
    ],
    "evidence_handles": []
  }
}
```

`evidence_handles` remains empty because the resolver observation is already
represented by the action's typed `source_refs`; this fix must not invent an
LLM evidence handle.

### Why this production surface is justified

`cognition_resolver/loop.py` is the only production file that directly authors
these three manual `speak` actions. A project-wide source search found no other
manual production `speak` dictionary outside this resolver file; canonical
cognition and self-cognition actions already pass through the materializer that
attaches provenance.

One resolver-local helper is justified because the same required, validated
current-user provenance is used by three builders. It prevents three copies of
identity validation and keeps the invariant at the producer owner.

### Rejected alternatives

- Repair only `_pending_resume_speak_action_spec`: rejected because the same
  missing-field failure remains in user-input and terminal fallback branches.
- Supply `target_roles=[]`: rejected because L3 would lose direct-recipient
  identity, especially in group scenes.
- Copy provenance from the discarded cognition action: rejected because the
  fallback also serves silent final-cognition paths with no action to copy and
  would create hidden dependence on state the resolver deliberately clears.
- Import the private cognition-node `_action_target_role`: rejected because
  it crosses the resolver-to-node ownership direction and risks import
  coupling. The resolver's narrower invariant applies only to visible
  user-message fallbacks.
- Change the generic `ActionSpecV1` validator: deferred because
  `cognition_provenance` is intentionally optional on the modality-neutral
  action schema and the observed defect is confined to L3-bound resolver
  producers. Broadening the global schema would affect unrelated action
  consumers and tests without evidence that they need the field.
- Relax the L3 guard or infer the user inside L3: rejected because it would
  erase caller-owned addressee authority and recreate the role-copy failure
  the August contract removed.

## Scope And Change Direction

### In scope

- Add one resolver-local deterministic provenance constructor.
- Require a non-empty resolved current global user id for visible resolver
  fallback actions.
- Attach its exact provenance to all three resolver-authored `speak` builders.
- Pass persona state into the two builders that currently lack it.
- Add direct producer tests, all-branch flow assertions, and one L3 handoff
  assertion.
- Update the resolver ICD and source/test ownership manifest.
- Restart the candidate brain after deterministic acceptance and replay the
  exact captured message with a fresh identity.

### Out of scope

- Semantic progression, memory pressure, response-goal selection, prompt text,
  model routing, provider behavior, or character wording.
- Action-spec schema/version changes or global provenance validation.
- L3, dialog, persona graph, adapter, service, database, pending-resume schema,
  emotion, relationship, or Conversation Progress production changes.
- Database cleanup, memory deletion, profile reset, or reuse of the failed
  identity.
- Any compatibility shim, feature flag, fallback mapper, retry, or new helper
  module.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Resolver visible fallback actions | bigbang | All three builders emit the one canonical provenance shape in the same edit. |
| Action schema and L3 consumer | keep | Preserve the existing generic validator and strict L3 guard. |
| Persistence | keep | Preserve all existing rows, schemas, and lifecycle behavior. |
| Tests | bigbang | Strengthen all three branch tests and add the exact owner contract nodes. |

## Mandatory Skills And Rules

- `development-plan`: maintain this separate lifecycle contract and evidence.
- `local-llm-architecture`: preserve semantic/deterministic ownership and zero
  model-call overhead.
- `py-style`: govern every Python edit and review.
- `cjk-safety`: preserve the existing CJK strings in `loop.py` and tests and
  run immediate syntax checks after edits.
- `test-style-and-execution`: use deterministic tests for this contract and run
  the real service case separately with inspected evidence.
- Preserve all concurrent working-tree changes. `loop.py`, its resolver README,
  and both resolver test files were clean at baseline. The manifest and plan
  registry already contained unrelated approved-plan edits and must be amended
  without replacing them.

## Must Do

1. Add deterministic RED assertions before production changes.
2. Prove pending-resume, user-input-blocker, and terminal-blocker actions all
   contain the exact current-user provenance.
3. Prove a resolver-produced fallback reaches the existing L3 addressee reader
   without an invariant error and projects `direct_recipient`.
4. Keep internal-thought/non-user sources private exactly as before.
5. Keep action delivery target, source refs, continuation refs, surface roles,
   reasons, content requirements, resolver statuses, and pending lifecycle
   unchanged.
6. Add zero LLM calls, retries, database calls, or prompt tokens.
7. Replay the captured service input with a fresh identity after restart and
   inspect the complete trace before resuming the original gates.

## Deferred

- Any judgment about whether the resulting Asuna reply passes the semantic
  progression rubric remains owned by the active progression plan.
- A broader action-spec provenance schema may be proposed only after evidence
  of a non-resolver producer gap.
- Memory deletion or correction remains a separate user decision after the
  complete progression evidence exists.

## Target State

```text
resolver chooses a user-visible fallback
  -> resolver validates current global user identity
  -> resolver constructs typed current-user cognition provenance
  -> resolver-authored speak passes generic action validation
  -> persona route selects text surface
  -> L3 reads exact caller-owned target role
  -> L3 projects current_user / direct_recipient
  -> dialog renders the selected resolver fallback
```

The change adds only bounded dictionary construction and validation. Runtime
LLM calls, database operations, resolver cycles, and prompt size are exactly
unchanged.

## Execution Roles

### Architect and sign-off owner

- Responsibility: own analysis, plan scope, architecture decisions, evidence
  review, and final sign-off.
- Owned surface: this plan, registry lifecycle, read-only review of all changed
  files and artifacts.
- Authority: approve or reject Luna output against this contract; request
  remediation inside scope; stop on scope expansion.
- Applicable skills: `development-plan`, `local-llm-architecture`,
  `test-style-and-execution`.
- Capability floor: system-level contract analysis and independent diff/test
  review.
- Independence requirement: must not author the production implementation.
- Acceptance output: reviewed diff, test evidence, live trace judgment, and
  lifecycle decision.
- Gate: begins after this complete plan exists; exits only when every
  acceptance criterion is evidenced.

### Production and test executor

- Responsibility: create RED tests, implement the resolver-only repair, run
  deterministic verification, restart the candidate service, and execute the
  disclosed live reproduction.
- Owned surface:
  `src/kazusa_ai_chatbot/cognition_resolver/loop.py`,
  `src/kazusa_ai_chatbot/cognition_resolver/README.md`,
  `tests/test_cognition_resolver_loop.py`,
  `tests/unit/cognition_resolver/test_loop.py`, and the exact resolver-loop row
  in `tests/ownership/source_test_impact_manifest.json`.
- Authority: edit only the listed surface, operate the candidate brain process,
  and create test artifacts; no DB cleanup or unrelated code edits.
- Applicable skills: `py-style`, `cjk-safety`,
  `test-style-and-execution`, and `local-llm-architecture`.
- Capability floor: production Python, async state-machine testing, Windows
  service control, exact artifact inspection, and dirty-worktree preservation.
- Independence requirement: separate from final sign-off owner.
- Acceptance output: scoped diff, RED/GREEN commands, exact collected nodes,
  live artifacts, and factual handoff.
- Gate: starts after architect handoff with captured baseline; exits after all
  owned tests and the fresh service reproduction pass.
- Plan-scoped fixed execution constraint: fresh GPT-5.6 Luna, maximum reasoning,
  normal speed, as explicitly required by the user. Changing this constraint
  requires the user's approval.

## Test Impact And Traceability

| Path | Changed contract | Owner | Exact deterministic pytest nodes | Supplemental node | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- | --- |
| `src/kazusa_ai_chatbot/cognition_resolver/loop.py` | Resolver-authored visible fallback provenance and current-user validation | Cognition resolver | `tests/unit/cognition_resolver/test_loop.py::test_resolver_surface_provenance_targets_current_user`; `tests/unit/cognition_resolver/test_loop.py::test_resolver_surface_provenance_requires_current_user`; `tests/test_cognition_resolver_loop.py::test_hil_repeated_after_pending_surfaces_pending_question`; `tests/test_cognition_resolver_loop.py::test_user_input_blocker_converges_after_one_final_cognition`; `tests/test_cognition_resolver_loop.py::test_duplicate_final_cognition_repeated_request_gets_terminal_speak` | `tests/test_semantic_response_progression_live_llm.py::test_live_l2_private_memory_enabled_theme_release` | Pure deterministic, patched graph handoff, then separately inspected real service | Any resolver fallback reaches L3 without required target provenance. |
| `src/kazusa_ai_chatbot/cognition_resolver/README.md` | Document current-user target provenance on visible fallback actions | Cognition resolver ICD | `tests/unit/cognition_resolver/test_loop.py::test_resolver_surface_provenance_targets_current_user` | none | Documentation-backed deterministic contract | Resolver documentation drifts from runtime ownership. |
| `tests/ownership/source_test_impact_manifest.json` | Map the changed loop owner to exact behavioral unit nodes | Test-impact registry | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`; `tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed` | `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run` | Deterministic manifest validation | A broad pass hides an uncollected or stale resolver owner test. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
  - Add the resolver-local provenance constructor.
  - Pass state to every manual visible fallback builder.
  - Attach exact provenance before `validate_action_spec(...)`.
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
  - State the resolver's visible fallback target-provenance contract.
- `tests/unit/cognition_resolver/test_loop.py`
  - Add direct exact helper contract and missing-identity failure tests.
- `tests/test_cognition_resolver_loop.py`
  - Strengthen all three fallback branch tests and exercise the L3 target-role
    handoff on the captured pending-resume shape.
- `tests/ownership/source_test_impact_manifest.json`
  - Add the new exact loop-owner nodes without replacing concurrent mappings.
- `development_plans/README.md`
  - Register and later close this plan without altering other rows.
- This plan
  - Record checkpoints, evidence, review, and lifecycle state.

### Keep

- `src/kazusa_ai_chatbot/action_spec/models.py` and all action-spec schemas.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` and its strict
  provenance guard.
- Persona graph, prompts, LLM configurations, dialog, adapters, service API,
  database, memory, Conversation Progress, emotion, and relationship owners.

### Create or delete

- None outside this plan file and normal ignored test artifacts.

## Agent Autonomy Boundaries

The Luna executor may choose local variable names, test assertion placement,
and command order while preserving the exact helper semantics, path list, and
acceptance nodes. It may not change action schema, L3, prompts, graph routes,
database behavior, memory, adapter behavior, or any file outside the declared
surface. Any required outside edit pauses execution for architect and user
review.

## Verification

### Baseline and RED

1. Confirm the captured hashes and dirty-state ownership recorded below.
2. Add the exact tests before production code.
3. Run the five mapped resolver nodes and confirm provenance assertions fail
   for the existing builders while unrelated assertions remain stable.

### GREEN deterministic checks

1. Run the two direct unit nodes.
2. Run the three branch nodes separately or in one deterministic batch.
3. Run the complete `tests/test_cognition_resolver_loop.py` suite.
4. Run the L3 target-addressee owner node and internal-source privacy nodes.
5. Collect every exact matrix node and run the manifest-backed impact command.
6. Run `py_compile`, CJK-safe AST parsing, and `git diff --check`.
7. Inspect the production diff for zero prompt/model/database/adapter changes.

### Real service gate

After deterministic acceptance, restart the candidate brain so it loads the
new source. Use a new channel/user identity and disclose the exact message
before sending:

```text
我最近工作有点乱，你会怎么帮我排第一步？
```

Pass for this plan:

- the request has no operational error;
- the resolver fallback action contains the exact current-user provenance;
- L3 and dialog both run;
- the visible response is delivered to the current user;
- no raw identifier appears in model-facing L3 input or visible output;
- no extra LLM call, resolver cycle, or DB operation is introduced by the fix.

Semantic quality, tactic release, character voice, and memory influence are
recorded but adjudicated by the separate semantic-progression plan.

## Acceptance Criteria

This plan is complete only when:

1. All three resolver-authored visible builders attach the same exact
   current-user provenance.
2. Missing current-user identity fails closed before L3.
3. Existing action source refs, targets, continuation refs, surface roles,
   reasons, content requirements, and resolver lifecycle results remain exact.
4. Non-user/internal cognition still produces no fabricated visible fallback.
5. L3 accepts a real resolver-produced action and projects the current user as
   `direct_recipient`.
6. Every exact mapped deterministic node collects and passes.
7. The complete resolver loop suite, manifest validation, syntax, and diff
   checks pass.
8. The fresh service reproduction reaches L3/dialog without the captured
   invariant error.
9. Independent parent review finds no scope expansion, compatibility layer,
   prompt/model change, memory behavior change, or unplanned production edit.
10. The active semantic-progression gate resumes only after this plan passes.

## Progress Checklist

- [x] Capture the real service failure and protected trace.
- [x] Localize all affected resolver-authored visible action builders.
- [x] Compare June producer history with the August L3 contract change.
- [x] Prove the semantic-progression diff did not modify this path.
- [x] Complete system ownership, alternative, overhead, and blast-radius
  analysis.
- [x] Receive explicit user approval for a separate plan and implementation.
- [x] Record deterministic RED evidence.
- [x] Implement the scoped resolver producer repair.
- [x] Complete mapped deterministic, adjacent, manifest, and static checks.
- [x] Complete fresh real-service reproduction and trace review.
- [x] Complete independent parent diff/evidence review.
- [x] Archive the completed plan and resume the semantic-progression gates.

## Execution Baseline

Captured before executor handoff on 2026-08-24:

```text
CB1097F34B54A0BAEEC53C9066E220F22EE058200AD96180B9B60E05A4384BAD  src/kazusa_ai_chatbot/cognition_resolver/loop.py
D07F04BA34E13917C637C45786BEEECF3B905750623132D74BA6ED6014E1A9D7  src/kazusa_ai_chatbot/cognition_resolver/README.md
F8860BF48B3B523C5D02986C93A948F589DC61545A353E1FF346AD35DC533AC8  tests/test_cognition_resolver_loop.py
61C33E212DB8D32505335B858B8FC1F112A433F23D9410532608D77B3BEB4B2C  tests/unit/cognition_resolver/test_loop.py
EAE53A3E0295E4843AC61490292C751B045666A3B29A2CFDCA998569CE0266BD  tests/ownership/source_test_impact_manifest.json
DDEDC33E753F6C494ABC5DFBC64FE82810AB28B8BA2055608D5EA7953E5E7219  development_plans/README.md
```

`loop.py`, the resolver README, and both resolver test files were clean.
The ownership manifest and plan registry were already modified by the active
semantic-progression and separate planning work. Their captured contents are
the merge baseline and must be preserved.

## Execution Evidence

- 2026-08-24: architect completed the system-level contract analysis and
  opened the approved plan for RED-first execution. The production boundary
  remains exactly `cognition_resolver/loop.py`; Luna owns the declared test,
  ICD, and manifest updates, while the parent owns independent review and
  sign-off.
- 2026-08-24: Luna added the five exact mapped assertions before editing
  production. All five failed on the missing helper or missing
  `cognition_provenance`, establishing RED evidence for the three producer
  branches.
- 2026-08-24: Luna implemented the one-file producer repair, updated the
  resolver ICD and exact manifest row, and reported 5/5 mapped nodes, 46/46
  resolver-loop tests, 3/3 L3/privacy checks, 2/2 manifest checks, and 52/52
  impact-selected tests passing. UTF-8 AST parsing, `py_compile`, and
  `git diff --check` also passed; only existing line-ending warnings remained.
- 2026-08-24: the independent architect inspected every production hunk and
  all call sites, required a precise manifest contract label, and reran the
  five mapped nodes, the full 46-test resolver suite, both manifest guards,
  UTF-8 AST parsing, `py_compile`, and scoped diff checks successfully. The
  review found one pure dictionary-construction helper, five state-argument
  call-site updates, and three provenance insertions, with no prompt, model,
  memory, schema, L3, dialog, database, adapter, relationship, or emotion
  production change.
- 2026-08-24: fresh real-service replay 1 used the exact captured message with
  new channel/user identity and memory enabled. It completed HTTP, L3, dialog,
  persistence, and background work without the prior invariant, but P selected
  `goal_resolution=answerable_now` with `resolver_requests=[]`. The strict
  fallback gate is `FAIL_NOT_EXERCISED`; the raw body was reconstructed from
  persisted conversation and trace evidence after the capture wrapper failed
  to write its response file. Evidence is under
  `test_artifacts/debug_runs/resolver_provenance_gate_20260824_service/turn_001/`
  with trace `llmtrace_56d881322e904f16beb71898fdb3200c`.
- 2026-08-24: one unchanged fresh-identity replay was run to avoid changing the
  failure case merely to force a branch. It captured the raw HTTP response and
  again completed normally, but P again selected `answerable_now` with no
  resolver request. The strict fallback gate remains `FAIL_NOT_EXERCISED`.
  Evidence is under
  `test_artifacts/debug_runs/resolver_provenance_gate_20260824_service/turn_002/`
  with trace `llmtrace_d8926ba3ed5a42c4b998326109585f6b`.
- Both fresh-user responses independently selected the same exchange-condition
  framing (`交易`, `报酬`, `条件`, or `交换条件`). This is retained as systemic
  evidence for the separate semantic-progression plan; it does not alter or
  satisfy this resolver provenance gate.
- 2026-08-24: after the unchanged captured input twice selected the normal
  answerable path, the user explicitly approved one branch-targeted message:
  `帮我继续处理刚才那个文件里的事项，先做哪一步？`. This changed only the live
  test stimulus so the same resolver failure mode could be exercised; it did
  not change the production contract or implementation scope.
- 2026-08-24: the approved fresh-identity turn selected
  `human_clarification` in both P passes, persisted the normal pending resume,
  detected the repeated capability, and exercised
  `_pending_resume_speak_action_spec`. The repaired action crossed the strict
  L3 provenance reader, and L3/dialog projected `current_user`, display name
  `Ren-Provenance-Gate-3`, and `direct_recipient`. All three assistant rows
  addressed resolved global user
  `d1e988fa-2ed3-4023-8294-4b95a90db441`; the prior invariant and raw-id
  leakage were absent. The 12 successful trace steps were the expected two
  cognition passes plus one L3 and one dialog call. Evidence is under
  `test_artifacts/debug_runs/resolver_provenance_gate_20260824_service/turn_003/`
  with trace `llmtrace_c3aec225f89d43699829a2f1e2b7f924`.
- 2026-08-24: independent parent review accepted the combined evidence
  boundary. Exact deterministic tests establish the one-user target-role and
  empty-evidence-handle dictionary for every producer; the live trace proves
  the real pending-resume producer crossed L3 and delivered to that resolved
  user. The repeated background `route_invalid` consolidation receipt matches
  the prior normal-path run and is outside this helper's behavior.
- Final sign-off: every acceptance criterion is satisfied. Production remained
  confined to `cognition_resolver/loop.py`; the repair adds deterministic
  dictionary construction only and preserves the hybrid agentic flow, model
  calls, memory, persistence lifecycle, emotions, relationship state, prompts,
  schemas, L3 ownership, dialog ownership, and adapter delivery contract.
