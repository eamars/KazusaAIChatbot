# dialog third-party target binding and addressee fidelity bugfix plan

## Summary

- Goal: preserve a named third-party target through cognition and dialog so a
  clause aimed at `蚝爹油` renders `蚝爹油` or an unambiguous third-person form,
  rather than using `你`, which the current dialog contract assigns to the
  immediate user.
- Status: completed
- Scope boundary: the ephemeral scene-participant role contract from
  decontextualization through goal cognition, L3 text surface, dialog
  verification, and focused evidence.
- Change direction: replace prose-only third-party identity with a bounded,
  prompt-safe participant handle and preserve the distinction between speech
  transport recipient and embedded semantic target.
- Acceptance state: implementation, deterministic verification, live evidence,
  quality review, and independent final review are complete; the plan is
  signed off for archival.

## Scope And Change Direction

The investigated turn is delivery tracking ID
`a81213ad11ee4f06a6ef63ee516c7bcc`. Its input correctly identifies `蚝爹油` as
the person who was frightened, but the accepted goal bid uses
`target_role_handles: ["current_user"]` while its concrete detail and
consequences describe `蚝爹油`. The final dialog therefore has two competing
signals: an explicit name and a fixed second-person rule that maps `你` to
`YCHDDZZ`.

The target behavior is fixed:

- `YCHDDZZ` remains the delivery and direct-message recipient for this turn.
- `蚝爹油` receives an ephemeral scene handle and remains the semantic target
  of the teasing, training, punishment, or control clause.
- The accepted final wording for that clause contains `蚝爹油` or an equally
  unambiguous third-person reference. It does not use `你` to mean
  `蚝爹油`.
- A turn whose semantic target is the current user continues to allow `你`.
- No deterministic post-generation replacement, regex, keyword gate, or
  character-specific text cleaner is introduced.

Ownership remains separated:

- Decontextualization owns semantic referent resolution from already visible
  scene participants.
- Cognition owns the semantic target and response goal.
- L3 surface and dialog own the visible addressee plan and wording.
- Deterministic code owns bounded role projection, exact contracts, prompt-safe
  identity redaction, validation, and retry limits.
- Conversation progress, consolidation, memory, and delivery observe their
  existing typed inputs and do not repair dialog targets after the fact.

Included work covers the missing participant binding, its typed propagation,
goal-target validation guidance, L3 addressee projection, dialog verifier
grounding, and regression evidence. Excluded work covers database migrations,
generic durable multi-user memory redesign, adapter delivery changes, and the
separate relationship-provenance finding recorded below.

## Evidence Baseline

### Observed facts

- The protected trace for the delivery contains a successful turn with one
  final dialog. The canonical trace is
  `llmtrace_87d3147ae14a42f4acc8fa046a165907` and the cognition failure capsule
  is `63b5a7abf4554e4db272c175ce7fcfd2`.
- The decontextualized percept preserves the third party in
  `role_explicit_content` and in `response_operation.operation`; its
  structured role vocabulary remains coarse (`其他参与者` and `无`).
- The goal prompt exposes only `current_user`, `r1`, and `self` as applicable
  handles for this turn. The accepted goal output retains
  `target_role_handles: ["current_user"]` while naming `蚝爹油` in its
  intention, concrete detail, and expected consequences.
- State projection binds `current_user` as the current user's target and has no
  typed handle for `蚝爹油`. L3 reduces target bindings to generic role labels
  and does not retain the visible participant name in the target projection.
- The dialog candidate role frame declares the current character as first
  person and the current user as second person. The generator prompt repeats
  that second-person rule and permits compatible creative expansion.
- The role-direction verifier is skipped when `selection_required` is false.
  Semantic fidelity receives the selected intent, content plan,
  requirements, and visible boundaries, but not the structured `addressee_plan`.
  The candidate consequently passed the existing verifier path on its first
  attempt.
- Conversation progress recorded the actor, action, object, and precondition
  with `蚝爹油`; it did not record the current user as the event target.
- The investigated exports contain zero current-user memory-unit rows and
  zero current-turn shared-memory rows. The residue contains an untyped `他`,
  so the short-lived residue remains ambiguous even though the durable event is
  correctly named.

Evidence files:

- `test_artifacts/diagnostics/llm_trace_llmtrace_87d3147ae14a42f4acc8fa046a165907_20260808T045819Z.json`
- `test_artifacts/diagnostics/turn_a812_progress_state.json`
- `test_artifacts/diagnostics/turn_a812_residue.json`
- `test_artifacts/diagnostics/turn_a812_user_memory_units.json`
- `test_artifacts/diagnostics/turn_a812_shared_memory.json`
- `test_artifacts/diagnostics/turn_a812_user_profile.json`
- `test_artifacts/diagnostics/turn_a812_lifecycle_records.json`
- `test_artifacts/diagnostics/turn_a812_all_events.json`

### Causal inference

The independent evidence review establishes this causal chain:

```text
named third party remains prose-only
  -> no typed third-party role handle exists
  -> current_user is accepted as the only usable target handle
  -> L3 reduces the target to generic "target"
  -> dialog applies second-person = current_user
  -> verifier accepts the ambiguous candidate
```

The high-confidence failure boundary is target binding, not memory
persistence. The target validator checks handle membership but does not check
semantic agreement between a selected handle and the named participant in the
goal prose. The first goal repair corrected relational-willingness structure
and retained the already-valid-but-wrong current-user handle.

### Separate persistence finding

The profile export also shows three relationship-axis increases backed by a
promoted-reflection reference. The available evidence does not establish that
this reflection belonged to the current user, and this change is independent
of the dialog target defect. This plan records the finding for a separate
relationship-appraisal provenance review; it does not change that subsystem.

## Confirmed Decisions

| Topic | Decision |
| --- | --- |
| Participant identity | Allocate deterministic, ephemeral handles such as `p1` from the already resolved visible participant roster. |
| Prompt identity | Expose the handle and exact visible display name; never expose platform IDs, global IDs, database IDs, or traces to the model. |
| Semantic target | Goal cognition selects the typed participant handle when the concrete action or relationship is about that participant. |
| Transport recipient | Keep the current user as the adapter delivery recipient even when the visible wording addresses or discusses another participant. |
| Surface authority | Carry the typed target and wording policy into the text-surface output and dialog verifier. |
| Verifier coverage | Existing semantic-fidelity verification owns the all-turn addressee check; the existing role-direction owner runs when a typed non-current addressee is present. |
| Persistence | Keep target handles transient. Progress and consolidation continue to make independent, source-bound persistence decisions. |
| Relationship profile | Defer the promoted-reflection relationship mutation to a separate provenance plan. |

## Mandatory Skills

- `development-plan`: lifecycle, scope, ownership, approval, evidence, and
  closeout gates.
- `local-llm-architecture`: semantic ownership, prompt projections, local-model
  reliability, and bounded call-radius decisions.
- `py-style`: all Python production and test changes.
- `cjk-safety`: Python prompt or fixture changes containing Chinese text.
- `test-style-and-execution`: deterministic tests, live LLM case execution,
  output inspection, and guarded database tests.
- `debug-llm`: human-readable quality review from captured raw evidence.
- `llm-trace-debug`: protected trace retrieval, raw prompt/output review, and
  trace evidence preservation.
- `character-test`: one-at-a-time real character-path replay and persistence
  inspection.
- `database-data-pull`: read-only export of progress, residue, profile, memory,
  and lifecycle evidence when the verification run needs fresh database data.

## Mandatory Rules

- Production implementation requires a separate explicit user command and an
  approved or in-progress lifecycle plan. This in-progress plan is executing
  under the user's explicit implementation command.
- Record `git status --short`, the current commit, Python version, governing
  documentation, and the exact source/test surface before implementation.
  Preserve unrelated worktree changes and never read `.env`.
- Use `venv\Scripts\python` for Python commands and `apply_patch` for manual
  edits.
- Keep semantic decisions in LLM-owned cognition and surface stages. Keep
  identity projection, exact shape validation, prompt redaction, limits,
  persistence, and delivery deterministic.
- Do not repair the generated dialog with string replacement, regular
  expressions, keyword lists, bracket removal, or a character-specific rule.
- Do not make the current user recipient and the embedded third-party target
  share one generic target field.
- Do not copy goal target handles into user memory, shared memory, profile
  updates, or conversation-progress writes as a persistence shortcut.
- Do not add a runtime database read, database collection, migration,
  compatibility alias, fallback role vocabulary, unbounded participant roster,
  or alternate dialog path.
- Preserve the canonical JSON parser, bounded regeneration behavior, existing
  delivery recipient fields, and source-bound consolidation gates.
- Run deterministic tests in batches. Run each live LLM case one at a time and
  inspect its output and captured prompts before accepting it.

## Must Do

- Define a prompt-safe `SceneParticipantBindingV1` contract containing an
  ephemeral handle, exact visible display name, and `third_party` entity kind.
  Bindings come only from the existing resolved scene roster and are scoped to
  one cognitive episode.
- Project non-current scene participants into the decontextualizer input as an
  allowed handle roster. Extend resolved referent rows with the selected
  participant handle; reject unknown handles, mismatched display names, and
  participant handles on unresolved referents through the normal bounded
  contract-retry path.
- Carry the binding roster into `SceneContextV2` and the cognition prompt
  projection. Add each binding to the allowed role-handle set with a prompt
  summary such as `p1=蚝爹油（群聊其他参与者）` while retaining no persistent
  identity in prompt payloads.
- Update goal cognition instructions and validation-facing feedback so
  `current_user` represents the observer/direct interlocutor and a named
  participant handle represents a concrete action or relationship target. The
  accepted bid for the investigated topology must use the participant handle
  for the teasing target while preserving the current user in expected
  observer consequences.
- Carry prompt-safe typed target projections from the admitted bid through
  `TextSurfaceInputV2`, L3 surface planning, and `TextSurfaceOutputV2`. Replace
  generic target-only summaries with a structured addressee/subject projection
  that contains the handle, visible display name, semantic role, and wording
  policy.
- Make the wording policy explicit: second person is allowed for the current
  user only when the current user is the intended clause target; a typed
  third-party target requires its visible name or an unambiguous third-person
  expression.
- Update dialog generation so the candidate role frame is derived from the
  authoritative typed target projection rather than treating every second
  person as the current user. Preserve natural direct address to the current
  user when that remains a separate speech recipient.
- Include the structured addressee/subject projection in semantic-fidelity
  verification. Run the existing role-direction verifier for typed
  non-current addressee cases, including non-selection turns, and reject a
  uniquely current-user-directed second-person clause when the authoritative
  clause target is the third-party handle.
- Preserve the accepted transport result: the output row remains addressed to
  `YCHDDZZ`; only the semantic wording target changes.
- Add focused deterministic tests, one-at-a-time live cases, protected trace
  evidence, and an agent-authored Markdown quality review. Verify the exact
  investigated topology, a direct-current-user case where `你` remains valid,
  and a no-third-party case that retains existing behavior.
- Verify progress, residue, user-memory, shared-memory, lifecycle, and profile
  evidence after the live case. Confirm that no wrong current-user durable
  memory is created, record the residue ambiguity if it remains, and report the
  separate relationship-provenance finding without folding it into this fix.
- Update the Cognition Core V2 and nodes ICD/README material with the
  participant-binding, transport-recipient, and dialog-addressee ownership
  boundaries.

## Deferred

- Fixing or migrating the promoted-reflection relationship mutation shown in
  `turn_a812_user_profile.json`; this requires a separate provenance plan.
- Persisting third-party handles, participant packets, or generic group
  relationship memory.
- Rewriting historical dialog, residue, conversation-progress events, profile
  rows, shared memory, or user memory.
- Adding deterministic semantic classifiers, name substitution, output
  sanitizers, or post-generation rejection outside the existing LLM verifier
  contract.
- Changing adapter mention syntax, delivery routing, group settlement,
  Conversation Progress ownership, consolidation lane eligibility, or memory
  extraction semantics.
- Adding an independent extra LLM judge beyond the existing semantic-fidelity
  and role-direction owners.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Decontextualizer referent contract | bigbang | Add the exact participant-handle field and update all callers, validators, prompts, and fixtures together. |
| Cognition role projection | bigbang | Use the ephemeral participant binding as the only typed third-party role path; do not retain a prose-only compatibility path. |
| L3 target/addressee contract | bigbang | Replace generic target-only prompt projection with the structured target and wording-policy projection in one contract update. |
| Dialog generation and verification | bigbang | Derive role direction from the authoritative target projection and verify non-current targets on the existing bounded path. |
| Transport delivery | compatible | Preserve the current-user addressed ID and adapter response shape. |
| Persistence and historical data | compatible | Make no migration or backfill; transient handles never enter durable collections. |
| Tests and fixtures | bigbang | Replace assertions that treat every `你` as a valid current-user target with the typed addressee contract. |

Rollback is a source revert. No persistent schema or historical row changes
are introduced.

## Target State

```text
resolved visible scene roster
  -> deterministic prompt-safe p1 = 蚝爹油 binding
  -> decontextualized referent carries p1
  -> cognition role handles include current_user, p1, and self
  -> goal bid targets p1 for the embedded teasing/control clause
  -> L3 preserves p1 name and emits a named/third-person wording policy
  -> dialog renders the current-user recipient and p1 semantic target separately
  -> semantic fidelity and role direction verify the same typed target contract
  -> delivery still addresses current user
  -> progress and consolidation retain independent source-bound persistence
```

## Contracts And Data Shapes

The implementation must use one canonical shape set:

```python
class SceneParticipantBindingV1(TypedDict):
    handle: str                 # p1, p2, ...; episode-local only
    display_name: str           # exact visible scene name
    entity_kind: Literal["third_party"]


class ReferentResolution(TypedDict, total=False):
    phrase: str
    referent_role: Literal["subject", "object", "time"]
    status: Literal["resolved", "unresolved"]
    participant_handle: str     # required for a resolved scene participant


class SurfaceAddresseePlanV1(TypedDict):
    handle: str
    display_name: str
    semantic_role: Literal[
        "direct_recipient",
        "embedded_target",
        "embedded_actor",
        "observer",
    ]
    wording_policy: Literal[
        "second_person_allowed",
        "named_or_third_person_required",
    ]
```

TypedDicts reside in their owning modules, and all callers consume these
canonical meanings. `entity_id` values used in internal role refs for an
episode-local participant must be non-persistent handles such as `scene:p1`;
platform IDs, global IDs, and database IDs never cross into an LLM payload.

`SceneContextV2` carries the bounded participant-binding list. The goal prompt
receives the same list as allowed role handles and summaries. The surface input
receives prompt-safe `SurfaceAddresseePlanV1` target context. The surface output
returns an exact `addressee_plan` list of those structured rows rather than
free-text-only addressee guidance. The dialog verifier receives the accepted
surface plan, the candidate role frame, the visible percept, and the candidate
dialog in one bounded payload.

The transport contract remains unchanged: `target_addressed_user_ids` and the
persisted conversation delivery row continue to identify the current user.

## Change Surface

### Delete

- None.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: extend the
  referent and episode-local participant contracts.
- `src/kazusa_ai_chatbot/nodes/referent_resolution.py`: normalize and validate
  the optional participant handle against the episode roster.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: derive the bounded
  prompt-safe participant roster from existing `scope_users` and carry it
  through persona state without exposing persistent IDs.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontextualizer.py`:
  expose allowed handles, require resolved names to retain their handle, and
  preserve the existing LLM-owned reference judgment.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: project the
  participant bindings into validated `SceneContextV2`.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: add the exact scene,
  role-target, and structured addressee contracts and validators.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`: export the new public
  participant and addressee contract types.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py` and
  `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`: expose episode-local
  third-party handles to goal cognition and preserve them through role
  bindings without making them mutable-state entities.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`: update target
  role guidance, prompt payloads, repair feedback, and contract handling for
  named third-party targets.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py` and
  `src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py`: carry and bound
  the prompt-safe target/addressee projection.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`: validate and
  preserve the structured addressee plan across preference and repair stages.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: preserve
  the admitted target identity and wording policy in the L3 handoff.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: derive the candidate role
  frame from the authoritative addressee plan, include that plan in semantic
  fidelity, and activate role-direction verification for typed non-current
  targets.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and
  `src/kazusa_ai_chatbot/nodes/README.md`: document target-vs-recipient
  ownership and the no-postprocessing dialog rule.
- `development_plans/README.md`: register this draft and update its lifecycle
  row only through the normal plan lifecycle.

### Create

- `tests/fixtures/dialog_third_party_target_fidelity_cases.json`: deterministic
  cases for the investigated group topology, direct current-user wording, and
  unresolved or absent third-party bindings.
- `tests/test_dialog_third_party_target_fidelity.py`: contract, projection,
  target-binding, verifier, and transport-recipient regressions.
- `tests/test_dialog_third_party_target_fidelity_live_llm.py`: individually
  runnable live dialog cases with captured stage inputs, outputs, and verdicts.
- `test_artifacts/diagnostics/dialog_third_party_target_fidelity/`: raw
  protected-trace exports and the parent-authored quality review for the live
  evidence.

### Keep

- `src/kazusa_ai_chatbot/conversation_progress/` event ownership and storage
  shape; its independent extraction remains the authority for progress facts.
- `src/kazusa_ai_chatbot/consolidation/` source policy, lane routing, memory
  eligibility, and persistence boundaries.
- Adapter delivery, current-user address fields, group settlement, RAG
  evidence, reflection promotion, and all existing durable collections.
- The canonical JSON parser and bounded LLM repair paths.

## Agent Autonomy Boundaries

The implementation owner may choose local helper names, function decomposition,
prompt serialization order, and focused test arrangement when those choices
preserve the contracts, ownership, and cutover policy above.

The implementation owner must not change the semantic authority, introduce
aliases or compatibility fields, preserve a prose-only fallback, add a
deterministic output rewrite, persist episode-local handles, or modify the
relationship-provenance subsystem. Any required change to transport delivery,
database shape, memory eligibility, LLM call ownership, or participant identity
source requires a plan amendment and user decision before implementation.

## Verification

Before implementation:

- Record `git status --short`, `git rev-parse HEAD`,
  `venv\Scripts\python --version`, the governing README/HOWTO files, and the
  exact source/test surface.
- Confirm no `.env` read and no unrelated worktree change is included.

Deterministic verification:

- Run the new focused contract and projection tests in a batch with the
  existing decontextualizer, cognition Core V2, surface, and dialog contract
  tests.
- Assert that prompt payloads contain only episode-local handles and visible
  names, never raw platform/global/database IDs.
- Assert that a typed p1 target produces a named/third-person wording policy,
  that a current-user target permits `你`, and that a missing typed target does
  not receive a fabricated participant binding.
- Assert that a uniquely current-user-directed candidate fails the typed p1
  target verifier and enters the existing bounded repair path; no deterministic
  text rewrite is called.
- Run `git diff --check` and repository anti-cheat searches for forbidden
  output-cleaning or captured-case substitution logic.

Live verification:

- Run the exact investigated group topology one case at a time through the
  character-test/debug path with a unique guarded test database scope.
- Capture the decontextualizer output, allowed participant roster, goal output,
  L3 surface input/output, dialog generator candidate, verifier verdicts, and
  protected trace in one review artifact. Metadata-only trace export is
  insufficient for this gate; the test harness must capture stage payloads in
  its approved diagnostic artifact.
- Require the investigated case to render the control clause with `蚝爹油` or
  an unambiguous third-person form, while delivery remains addressed to
  `YCHDDZZ`.
- Run a direct-current-user case and require natural `你` wording to remain
  accepted when the typed target is current user.
- Inspect progress, residue, user-memory, shared-memory, lifecycle, and profile
  exports. Require no current-user or shared-memory durable row attributable to
  the transient taunt; record any residue ambiguity as evidence.
- Record the promoted-reflection relationship mutation as a separate finding
  and do not use this plan's dialog result to declare that provenance issue
  fixed.

Review gates:

- Produce one parent-authored Markdown quality review that separates observed
  outputs, contract verdicts, and semantic quality judgment.
- Complete one independent code review against this plan's ownership,
  contracts, exclusions, and evidence. A review finding that expands target
  identity or persistence scope returns to plan amendment.

## Acceptance Criteria

The plan is accepted for implementation only when the user approves the draft
and issues an explicit implementation command. Implementation acceptance is
binary:

1. The investigated topology produces a typed third-party handle for
   `蚝爹油`; the accepted goal target for the embedded control clause is not
   `current_user`.
2. The handle and exact display name survive goal, surface, dialog, and
   verifier payload validation without persistent identity leakage.
3. A final candidate using `你` for the p1 control target is rejected or
   repaired by the existing bounded LLM-owned verifier path; the accepted
   candidate uses `蚝爹油` or an unambiguous third-person form.
4. A direct current-user target continues to accept `你` and remains delivered
   to the current user.
5. Conversation progress continues to record `蚝爹油` as the event object, and
   no current-user or shared-memory durable row is created for the one-off
   taunt.
6. No production path contains a deterministic dialog rewrite, compatibility
   alias, extra persistence lane, or unapproved database change.
7. The focused deterministic tests, one-at-a-time live evidence, parent
   quality review, and independent code review all pass with artifacts linked
   in the execution record.

## Progress Checklist

- [x] Read the development-plan skill, plan contract, cutover policy,
  execution gates, and development-plan registry.
- [x] Inspect current worktree, governing README/HOWTO, relevant subsystem
  ICDs, source, tests, and protected diagnostic exports.
- [x] Obtain independent evidence-only RCA from the requested GPT-5.6 Sol
  high-reasoning subagent.
- [x] Separate the dialog-target defect from the unrelated relationship-profile
  provenance finding.
- [x] User approves this draft and explicitly commands implementation.
- [x] Implementation updates the exact contract and test surface.
- [x] Focused deterministic verification passes.
- [x] One-at-a-time live replay and persistence evidence pass.
- [x] Independent code review passes and the user signs off the final evidence.

## Independent Plan Review

The independent RCA was performed by GPT-5.6 Sol with high reasoning effort in
an isolated subagent context (`Archimedes`, agent ID
`019fdfee-101c-78b0-bca4-37e013ef47d7`). Its authority was read-only evidence
analysis and fix guidance; it did not edit production code, alter the database,
or author this plan.

Its decision was to accept the concern about the final `你`, reject the claim
that the wrong target became a durable current-user memory consequence, and
qualify the separate relationship-profile mutation as an independent
provenance issue. The plan adopts those findings as the evidence baseline and
keeps the implementation boundary explicit.

## Execution Record

- Baseline commit: `06fd46228d50c97024b2a612a40f66a578907e4a`; Python:
  `3.14.6`. The pre-existing user changes remained preserved, including the
  separate cognition-core plan and its registry entry.
- The requested DeepSeek implementation handoff used the required
  acknowledgement turn and bounded execution turn. Hubble
  (`019fe07a-254c-7d32-9845-ed7a98560cc9`) timed out after the required
  600-second execution window with a partial contracts edit. The parent
  reviewed and completed the implementation, then closed the worker.
- The implementation adds bounded `p1` participant bindings, typed referent
  and addressee contracts, prompt-safe projections through cognition/L3/dialog,
  current-user transport separation, and verifier coverage. It does not add a
  persistence lane or deterministic text rewrite.
- Focused deterministic verification passed: 141 tests across the new
  fidelity cases and adjacent contract/dialog/decontextualizer/referent tests.
  The recorded broader impacted deterministic batch passed 206 tests with
  four deselections; a later superset rerun passed 338 tests with the same
  live-test exclusions. Python compilation and `git diff --check` passed.
- The one-at-a-time live stage cases passed for named third-party, direct
  current-user, and no-third-party topologies. The guarded full character path
  passed with the authoritative artifact
  `test_artifacts/diagnostics/dialog_third_party_target_fidelity/character_path_group_named_third_party__20260808T224053038925Z.json`.
  It records `蚝爹油` as `p1`, preserves current-user-only delivery, records the
  named event object in canonical progress, and produces zero residue,
  user-memory, and shared-memory rows for the replay.
- The parent-authored evidence review is
  `test_artifacts/diagnostics/dialog_third_party_target_fidelity/quality_review.md`.
  It separates observed facts, contract verdicts, semantic quality judgment,
  and the independently deferred relationship-profile provenance concern.
- The first independent DeepSeek final review returned `PASS` with low-severity
  findings. The evidence omission was documented in the quality review, and a
  deterministic p1 wrong-`你` repair-handoff regression was added and included
  in the 141-test pass. The structural projection override and harmless prompt
  noise were retained as documented non-blocking design notes. The subsequent
  final DeepSeek review also passed; its import-order and role-frame noise
  findings were corrected, its prompt-vocabulary note was addressed by using
  plain semantic wording, and the line-ending-only unrelated test change
  remains preserved.

## Final Sign-Off And Closeout

- The final independent DeepSeek reviewer, Turing
  (`019fe3a8-5c20-7160-9a3c-67767958b873`), returned `PASS` with no blocking
  findings and confirmed that the plan is ready for formal closure.
- The parent review confirms all seven acceptance criteria: typed p1 target
  continuity, prompt-safe projections, bounded wrong-`你` repair, current-user
  transport separation, persistence isolation, forbidden-mechanics exclusion,
  and complete evidence/review gates.
- The one remaining checklist gate is closed by this sign-off. The plan is
  marked `completed` and is ready to move from `active/bugfix/` to
  `archive/completed/bugfix/`. The separate user-authored cognition plan and
  registry entry remain unchanged.
