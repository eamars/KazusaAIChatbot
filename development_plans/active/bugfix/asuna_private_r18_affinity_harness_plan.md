# Asuna private R18 affinity full E2E harness

Status: completed

## Summary

Build a test-only full end-to-end harness for the exact twenty ordered user
inputs in
`test_artifacts/cognition_core_v2/private_r18_replay/replay_manifest.json`.
Run the same twenty inputs twice against the guarded
`_test_kazusa_live_llm` database with the adult Asuna profile. Each condition
uses one persistent service lifespan and sends every input through the public
`/chat` route.

The high-affinity condition starts with the manifest's native high relationship
axes. The default-affinity condition starts from a fresh empty baseline using
the canonical `build_acquaintance_user_state` relationship seed. The source
R18 artifact contributes only the twenty user messages and their typed input
envelopes; old dialog, residue, memory, and history rows are not copied.

## Scope and ownership

- `character-test`: inspect every live turn, response surface, trace, state
  transition, persistence effect, and delivery capture.
- `debug-llm`: retain raw trace evidence and produce the human-readable
  comparison review after the live artifacts exist.
- `database-data-pull`: treat the existing frozen manifest as read-only source
  input; no production database write path is used.
- `local-llm-architecture`: keep semantic acceptance/rejection judgment in the
  live LLM stages and keep the harness at the service boundary.
- `py-style`, `cjk-safety`, and `test-style-and-execution`: govern all Python
  edits and live-test execution.

No production source, prompt, personality field, schema, or adapter behavior
changes are in scope.

## Required behavior

1. Validate the manifest schema, source identity, exact twenty case indexes,
   and every source message before any live write.
2. Reset only `_test_kazusa_live_llm` and seed one native user identity plus
   the selected relationship state. Start with zero conversation, memory, and
   internal-residue rows.
3. Open one service lifespan per condition and call the public FastAPI
   `/chat` route exactly once for each input in chronological order.
4. Keep state in the same service/database session across all twenty turns.
   Record before/after native user state, full conversation rows, response,
   cognition graph, trace run, ordered trace steps, lifecycle record, Mongo
   counts, and captured adapter delivery for every turn.
5. Continue after bounded service-level failures that return an inspectable
   response and persisted trace. A missing response, missing trace, missing
   persisted input, HTTP route crash, or process crash is a fatal sequence
   failure and stops that condition.
6. Reset and reseed the empty baseline before the default-affinity sequence.
7. Write a side-by-side review containing all twenty inputs, both observed
   responses, technical status, trace status, state transitions, and artifact
   paths. Expected high-affinity engagement and default-affinity boundary are
   review targets, not deterministic keyword assertions.
8. Restore the guarded database to the empty default-affinity baseline after
   both sequences.

## Full E2E boundary

The child process enters the real service lifespan, registers the test-only
QQ delivery adapter, builds a real `ChatRequest`, serializes it as JSON, and
posts it through the public `/chat` FastAPI route using the app transport. The
route enters the production queue, RAG/cognition/dialog pipeline, persistence,
post-turn lifecycle, and adapter delivery path. One child process owns the
complete twenty-turn condition sequence; no turn is launched as an isolated
child test.

## Evidence contracts

Each turn artifact is an
`asuna_private_r18_affinity_e2e_turn.v1` object containing:

- exact source user message and public request envelope;
- HTTP response and parsed `ChatResponse` surface;
- complete native user state before and after the turn;
- response cognition graph, trace run, ordered trace steps, and dispositions;
- persisted user/assistant rows, lifecycle record, conversation sequence, and
  collection counts;
- captured adapter calls, continuity checks, duration, and technical status.

Each condition manifest is an
`asuna_private_r18_affinity_e2e_run.v2` object containing the source-input
ledger, empty-baseline seed, one-session execution contract, ordered artifact
paths, final state, and any fatal or bounded service failures.

## Change surface

- `development_plans/active/bugfix/asuna_private_r18_affinity_harness_plan.md`
- `tests/test_asuna_private_r18_affinity_live_llm.py`
- `tests/test_asuna_private_r18_affinity_harness_contract.py`
- `tests/run_asuna_private_r18_affinity_replay.py`
- `test_artifacts/cognition_core_v2/asuna_private_r18_affinity_replay/`
- `development_plans/README.md` registry entry.

## Execution order

1. Run collection, compilation, and deterministic contract tests.
2. Run the high-affinity child for all twenty turns and inspect each artifact
   and child log before the next condition.
3. Reset the guarded database and run the default-affinity child for all twenty
   turns, again inspecting every artifact.
4. Render the full comparison review.
5. Restore and verify the empty default-affinity baseline.
6. Verify the worktree contains no production-code changes.

## Guardrails

- The runner rejects any database other than `_test_kazusa_live_llm`.
- Output paths must remain below the replay artifact root.
- Background workers are explicitly disabled so state transitions remain
  inspectable and deterministic at the live-test boundary.
- The user-input projection does not read or seed the manifest's old dialog,
  residue, memory, or exported history fields.
- The test records model outcomes and technical failures without rewriting
  semantic decisions.

## Acceptance criteria

- Deterministic contracts pass.
- Twenty high-affinity and twenty default-affinity turn artifacts exist after
  successful execution.
- Each condition manifest proves one persistent service session and twenty
  ordered public `/chat` calls.
- Every completed turn contains inspectable input, response, trace, state, and
  persistence evidence, or a typed bounded service failure.
- A fatal crash stops the current condition and leaves a fatal artifact with
  traceback and the exact input that could not advance.
- The default run begins from a verified empty baseline, not high-affinity
  state.
- The comparison review contains the complete twenty-input side-by-side
  sequence and expected-behavior deviations.
- Final restoration and guarded-database/worktree checks pass.
