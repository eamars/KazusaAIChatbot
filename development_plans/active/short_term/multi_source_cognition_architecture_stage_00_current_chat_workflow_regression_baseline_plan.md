# multi source cognition architecture stage 00 current chat workflow regression baseline plan

## Summary

- Goal: Capture the current `/chat` workflow as a deterministic regression gate
  before any multi-source cognition refactor changes runtime contracts.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `cjk-safety` if Python tests add
  CJK strings.
- Overall cutover strategy: test-only additive change; no production source
  behavior may change.
- Highest-risk areas: accidental source edits, brittle exact LLM text snapshots,
  missing debug-mode coverage, and a baseline that does not cover RAG,
  cognition, dialog, persistence, and consolidation boundaries.
- Acceptance criteria: deterministic tests and fixtures exist, run locally, and
  can be reused by later stages as a hard no-regression gate.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_00`

## Context

The parent architecture requires `/chat` migration before reflection,
monologue, image, audio, or proactive triggers are enabled. This stage creates
the baseline that later stages must pass before they can proceed.

The baseline should not attempt to specify exact live LLM prose. It should lock
down deterministic contracts: graph route, state shape, required fields, prompt
renderability, persistence decisions, delivery tracking, debug-mode behavior,
and RAG projection shape.

## Mandatory Skills

- `development-plan-writing`: keep this child plan aligned with the parent
  architecture.
- `local-llm-architecture`: preserve bounded live-chat behavior and avoid
  brittle local-LLM assertions.
- `py-style`: load before editing Python test files.
- `test-style-and-execution`: load before adding or running tests.
- `cjk-safety`: load before editing Python test files that contain CJK strings.

## Mandatory Rules

- Do not edit production `src/` files in this stage.
- Do not edit existing test modules except for reading them; the stage-specific
  harness must live in the new test module listed in `Change Surface`.
- Do not add `CognitiveEpisode` or any runtime episode contract in this stage.
- Do not change prompts.
- Do not change database schemas.
- Do not use exact generated LLM prose as a deterministic assertion.
- Real LLM checks, if added, must be marked and run one at a time with output
  inspection.
- Mocked prompt-render checks are characterization tests. They may assert that
  prompt templates render, expected payload fields are present, and JSON parsing
  contracts are satisfied; they must not claim live prompt quality.
- No runtime LLM calls are added. No response-path latency change is allowed.
- If a production source change appears necessary, stop and create a separate
  bugfix plan.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.

## Must Do

- Add a focused deterministic baseline test module.
- Add reusable regression fixtures for representative `/chat` cases.
- Cover normal text response, cognition-selected silence, group history scoping,
  RAG skip shape, debug modes, assistant delivery tracking, and consolidation
  handoff shape.
- Include prompt-render checks for existing L1/L2/L3 and dialog prompt paths
  using mocked LLM responses.
- Record a frozen synthetic evidence corpus that later RAG adapter work can use
  for retrieval-equivalence tests.

## Deferred

- Runtime `CognitiveEpisode` implementation.
- Source-aware prompt selection.
- RAG adapter refactor.
- Reflection-triggered cognition.
- Internal thought, action latch, image, audio, and proactive output support.

## Cutover Policy

There is no production cutover. The only cutover is adding tests and fixtures.
The tests become a required preflight for later child plans.

## Agent Autonomy Boundaries

The implementation agent may choose test helper names and fixture structure, but
must keep them under the listed change surface. The agent must not broaden this
stage into source refactoring or prompt tuning.

## Target State

Later stages can run one command to prove the current `/chat` behavior still
holds. The baseline should be stable under mocked LLMs and should fail loudly
when graph routing, required fields, debug modes, persistence decisions, or
RAG-projection shape drift.

## Design Decisions

- Baseline exact interfaces, not exact generated prose.
- Use synthetic fixtures for stable deterministic coverage.
- Reuse existing tests where practical, but add one stage-specific test module
  so later plans have a single obvious regression target.
- Keep the frozen evidence corpus small and inspectable.

## Change Surface

### Create

- Add `tests/test_multi_source_cognition_stage_00_regression_baseline.py`.
- Add `tests/fixtures/multi_source_cognition_stage_00_cases.json`.

### Modify

- Update this child plan's progress checklist and `Execution Evidence`.
- Update
  `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  ledger status for `stage_00`.
- Update `development_plans/README.md` active-plan table status for `stage_00`.

### Keep

Small test-only helpers may be added inside the new stage-specific test module.

### Forbidden

- `src/kazusa_ai_chatbot/**`
- database schema files
- prompt source files
- adapter source files
- existing test modules, unless a verification failure proves the existing
  helper contract is unusable and the agent stops for a bugfix plan first.

## Implementation Order

1. Create the fixture file with case ids for private text, group text, reply,
   silence, RAG skip, RAG hit, `think_only`, `no_remember`, and `listen_only`.
   The same fixture file must include a top-level `frozen_evidence_corpus`
   object for future retrieval-equivalence tests.
2. Add graph-route tests around `persona_supervisor2` using patched
   decontextualizer, RAG, cognition, and dialog nodes.
   Patch these existing seams:
   `call_msg_decontexualizer`, `stage_1_research`,
   `call_cognition_subgraph`, and `dialog_agent`.
3. Add service-level debug-mode tests in the new stage module through
   `service.chat` with patched dependencies. Cover `think_only`,
   `no_remember`, and `listen_only`.
4. Add RAG projection shape tests using existing RAG skip/projector behavior.
5. Add prompt-render tests that patch LLM calls and assert required payload
   fields are rendered without template errors for:
   `call_cognition_subconscious`, `call_cognition_consciousness`,
   `call_boundary_core_agent`, `call_judgment_core_agent`,
   `call_contextual_agent`, `call_style_agent`,
   `call_content_anchor_agent`, `call_preference_adapter`,
   `call_visual_agent`, `dialog_generator`, and `dialog_evaluator`.
6. Add a frozen synthetic evidence corpus section to the fixture for future RAG
   adapter equivalence.
7. Run the verification commands and record results in the plan if the plan is
   later executed.

## Progress Checklist

- [x] Regression fixture file created.
  - Sign-off: Codex / 2026-05-09 after targeted baseline verification.
- [x] Persona graph route tests added.
  - Sign-off: Codex / 2026-05-09 after targeted baseline verification.
- [x] Debug-mode behavior covered.
  - Sign-off: Codex / 2026-05-09 after targeted baseline verification.
- [x] RAG shape covered.
  - Sign-off: Codex / 2026-05-09 after targeted baseline verification.
- [x] Prompt-render checks covered.
  - Sign-off: Codex / 2026-05-09 after targeted baseline verification.
- [x] Frozen evidence corpus created.
  - Sign-off: Codex / 2026-05-09 after targeted baseline verification.
- [x] Deterministic verification passed.
  - Sign-off: Codex / 2026-05-09 after required and optional verification.
- [x] Completion artifacts recorded in `Execution Evidence`.
  - Sign-off: Codex / 2026-05-09.
- [x] Parent ledger updated with `stage_00` completion status.
  - Sign-off: Codex / 2026-05-09.

## Verification

Required deterministic verification:

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py tests\test_persona_supervisor2_rag_skip_shape.py
```

Optional broader deterministic verification:

```powershell
venv\Scripts\python -m pytest tests\test_dialog_agent.py tests\test_consolidator_efficiency.py tests\test_cognition_interaction_style_context.py
```

Real LLM smoke tests are not required for this stage. If added later, run one
case at a time with `-s` and inspect the output before continuing.

## Acceptance Criteria

- The stage-specific baseline test module exists and passes.
- The fixture file exists and is small enough to inspect manually.
- No production source files changed.
- No prompt semantics changed.
- Later child plans can cite this stage as their hard `/chat` regression gate.
- Parent ledger can point to this stage's test module, fixture file, and
  verification evidence.

## Completion Artifact Contract

`stage_00` is not complete until `Execution Evidence` records:

- The regression test module path.
- The regression fixture path.
- The frozen evidence corpus location inside the fixture.
- The exact deterministic commands run and their result.
- Any skipped live LLM smoke checks and the reason they were skipped.
- Confirmation that `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  was updated so `stage_00` is no longer `draft` in the parent ledger.

## Risks

- The baseline may miss an important live path if fixtures are too narrow.
- Overly broad tests may become slow and block iteration.
- Exact LLM-output snapshots would become noisy and should be avoided.

## LLM Call And Context Budget

No runtime LLM calls are added. Deterministic tests must mock LLM calls. The
response-path call count, prompt text, prompt payload shape, and context budget
remain unchanged because no production source files are modified.

Prompt-render tests may execute existing handler code with fake LLM objects to
capture rendered messages. These tests are deterministic characterization
checks only. Any future live smoke case must remain outside the default
deterministic command and must be run one case at a time with output
inspection.

## Execution Evidence

- Regression test module:
  `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
- Regression fixture path:
  `tests/fixtures/multi_source_cognition_stage_00_cases.json`
- Frozen evidence corpus location:
  `tests/fixtures/multi_source_cognition_stage_00_cases.json` at top-level
  key `frozen_evidence_corpus`.
- Deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py -q`
    passed: 11 passed.
  - `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_service_background_consolidation.py tests\test_persona_supervisor2_rag_skip_shape.py -q`
    passed: 26 passed.
- Optional broader deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_dialog_agent.py tests\test_consolidator_efficiency.py tests\test_cognition_interaction_style_context.py -q`
    passed: 17 passed.
- Additional branch-level verification outside required stage gates:
  - `venv\Scripts\python -m pytest -q` failed during collection in
    `tests/test_e2e_live_llm.py` and
    `tests/test_persona_supervisor2_rag_supervisor2_live.py` because both
    modules import `get_db` from `kazusa_ai_chatbot.db`.
  - `venv\Scripts\python -m pytest -q --ignore=tests\test_e2e_live_llm.py --ignore=tests\test_persona_supervisor2_rag_supervisor2_live.py`
    failed: 810 passed, 9 failed, 163 deselected. Failing tests were in
    existing character/user snapshot, memory writer, RAG initializer, and user
    state snapshot modules.
- Static checks:
  - `git diff --check` passed with only the existing CRLF warning for
    `development_plans/README.md`.
  - `git diff --name-only -- src` returned no files.
  - `git ls-files --others --exclude-standard src` returned no files.
- Live LLM smoke checks:
  skipped because this stage does not require real LLM smoke tests and all
  prompt-render checks use mocked LLMs.
- Parent ledger:
  `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  updated so `stage_00` is `completed`.
