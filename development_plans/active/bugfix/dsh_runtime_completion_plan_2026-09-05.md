# DSH runtime completion

## Goal and authority

- **Status:** in_progress.
- **Goal:** a natural request reaches the actual DSH model, obtains grounded
  evidence, returns through character cognition, and produces the correct
  visible answer or one deferred delivery. Failures leave truthful, recoverable
  state and release resources.
- **Scope:** the entire existing and new DSH codebase, including Python control
  plane, TypeScript sidecar, semantic tools, Brain interactions, cognition,
  accepted tasks, background work, persistence, and adapter delivery integration.
  Scope is independent of authorship, commit date, and the current git diff.
- **User direction:** remove ownership tests, remove dedicated DSH tests, cut
  80% of DSH integration unit tests, preserve every real LLM test, and replace
  test-focused planning with one plan that establishes a working runtime.
- **Acceptance:** runtime behavior remains unverified with the real model.
  Previous green deterministic runs establish only their exercised boundaries.

This is the sole active DSH execution contract. It supersedes the September 4
cleanup/sign-off and September 5 integration/test-remediation plans. Their
unfinished functional obligations and existing fixes carry forward; their
test matrices, per-source unit mandates, and non-live-before-live gates expire.
Archived records remain historical evidence, including failed attempts.

The user's current direction overrides the development-plan skill's mandatory
owner-unit mapping and supplementary-only status for real-model evidence.
Source ownership remains an architectural responsibility. It no longer creates
a test-per-file obligation or a source-to-test manifest gate.

## Test reduction contract

Use the complete pre-cut working tree as the baseline, including existing and
new tests. Evidence is in `test_artifacts/dsh_scope_reset_20260905/`.

1. Remove code-structure ownership checks: module/export discoverability,
   static package-boundary policing, and the manifest validator, manifest,
   validator tests, CLI registration, and current documentation requirements.
   Business permission, audience, lease, and record ownership are runtime
   concepts; a mention of ownership alone does not classify a test for deletion.
2. Remove every non-real-LLM dedicated suite under
   `tests/test_dsh_*`, `tests/test_agentic_resolver_*`,
   `tests/test_task_resolution_*`, `tests/unit/agentic_resolver/`,
   `tests/unit/task_resolution/`, `tests/integration/test_dsh_runtime_probe.py`,
   and all project-authored `sidecars/dsh_resolution/tests/` cases.
   This includes dedicated live-DB tests. The real LLM preservation rule takes
   precedence over every filename and deletion category.
3. Reduce the DSH cases embedded in other subsystems by at least 80%, measured
   by existing test function definitions, not files or parameter expansion.
   The reviewed selection contains 129 embedded cases: remove 104 and retain
   25 (80.62%). Within `tests/unit/`, remove 62 of 77 (80.52%). Incidental
   references in unrelated tests are excluded from that denominator.
4. Preserve all real LLM test definitions and collection identities, including
   the DSH behavior and task-result dialog cases. Preserve their required
   fixtures, diagnostic capture, and process/database cleanup support.
5. Delete helper code made unreachable by these cuts. Retain shared helpers
   required by surviving tests and live cases. Repair imports and collection;
   do not rename or relocate deleted cases to recreate the removed suite.

`cut_decisions.json` records each baseline test's disposition and reason.
`test_hashes_before.json` and `live_collection_before.log` protect existing
content and real LLM collection. This is a one-time deletion record, not a new
enforced ownership manifest. Completion requires the actual before/after
counts and unchanged live collection, rather than a new permanent meta-test.

## Execute the real path first

### 1. Minimal readiness, then foreground model probe

Build only what is needed to launch the actual sidecar. Establish a guarded
temporary database/workspace, configured model route, authenticated readiness,
and owned-process cleanup. Start the smallest existing foreground real-model
case immediately. Full regression suites, coverage, owner matrices, and an
independent readiness campaign are not prerequisites.

```powershell
$env:KAZUSA_RUN_LIVE_LLM='1'
venv\Scripts\python -m pytest tests/test_dsh_behavior_live_llm.py::test_live_foreground_task_resolution_is_grounded_and_character_owned -m live_llm -q -s
```

The two release notes contain different owners/prerequisites. The first input
is ambiguous; the second specifies Release B. Inspect whether the character
handles ambiguity and then uses actual DSH evidence to identify Mira and the
pending checksum review. Accept paraphrase and valid tool-order variation.
Require real task entry, source grounding, visible output, correlated traces,
and cleanup. A plausible answer without DSH execution fails this probe.

If it fails, classify the observed failure as harness, configuration, transport,
DSH execution, semantic judgment, persistence, or visible rendering. Preserve
raw input/output, traces and state before cleanup. Change only the demonstrated
boundary and rerun that case. Pause test expansion and cosmetic hardening until
the real path works. An unavailable model is an explicit unverified boundary.

### 2. Deferred completion and internal judgment

Run these existing cases separately. Inspect and judge each result before
starting the next:

```powershell
venv\Scripts\python -m pytest tests/test_dsh_behavior_live_llm.py::test_live_deferred_task_result_recurs_and_delivers_once -m live_llm -q -s
venv\Scripts\python -m pytest tests/test_dsh_behavior_live_llm.py::test_live_internal_dsh_judgment_is_character_owned -m live_llm -q -s
```

Deferred success means the actual delivered comparison reports the supported
changes, Rowan, and the missing threshold, through one accepted-task/job lineage
and one eligible delivery. Internal success means an answerable signed question
uses the available evidence and an unsupported success claim receives no
approval. Inspect decision reasoning as well as enum validity.

Preserved live dialog cases remain runnable. Execute the specific case affected
by a demonstrated rendering defect; preserving all live cases does not mandate
running the repository's unrelated live campaign.

### 3. Exercise failure recovery, then harden

After model viability, use the existing executable diagnostic entry point
`experiments/dsh_runtime_probe.py` to inspect `sidecar-lifecycle`,
`brain-task-lifecycle`, and `transport-loss`, each with a fresh artifact directory.
These commands exercise real processes/storage with a controlled provider;
their results establish recovery mechanics, not model quality. Preserve the
CLI scenarios as diagnostic tools; remove their duplicate pytest wrappers.

For a cancellation or promotion failure observed during execution, use the
public task/runtime interface and owned process/store to reproduce the exact
race. Require committed terminal/background ownership to survive, unowned work
to settle, and uncertainty to remain typed and auditable. Avoid rebuilding the
deleted simulated lifecycle suite.

Run the 25 retained integration cases after viability, plus scoped Python
syntax/lint and sidecar strict typecheck/build. Broaden checks only when an
observed failure or a changed shared boundary justifies them. Repeat the
affected real probe after any production fix or hardening change.

## Change surface and quality boundaries

Production diagnosis and evidence-driven correction cover `src/agentic_resolver/`,
`sidecars/dsh_resolution/src/`, `src/kazusa_ai_chatbot/task_resolution/`,
`dsh_interaction/`, and `dsh_tool_gateway/`, plus their concrete callers under
`cognition_core_v3/`, `cognition_resolver/`, `cognition_episode/`, `nodes/`,
`accepted_task/`, `background_work/`, `brain_service/`, `db/`, and `service.py`.
The latter paths are relative to `src/kazusa_ai_chatbot/`. Adapter/console
changes are limited to demonstrated DSH launch or delivery integration defects.

Maintain the existing canonical contracts: RAG supplies evidence, cognition
judges and decides, dialog renders, deterministic code validates and persists.
Keep asynchronous cancellation, one timeout owner, idempotent operations,
lease/generation fencing, bounded repair/regeneration, audience isolation,
secret isolation, and one eligible delivery. Preserve required-data fail-fast
behavior. Repairs use existing public interfaces and semantic stage owners.
Avoid compatibility layers, duplicated semantic state, test-only production
hooks, and keyword-based rewrites of LLM decisions.

Test-removal support includes `tests/`, the sidecar test/build configuration,
`scripts/validate_test_impact.py`, `pyproject.toml`, current README/HOWTO guidance,
and this plan/registry. Preserve unrelated business behavior and current
production edits. Use the project venv and exact temporary resource guards.
Deployment, production database changes, environment-file inspection, unrelated
architecture redesign, and new features remain outside this work.

## Runtime evidence and test impact

| Boundary | Acceptance evidence | Failure detected |
|---|---|---|
| Brain/cognition -> task service -> Python runtime -> Standard sidecar -> model/tools -> result -> dialog | Exact foreground live node above; full request, model traffic, task evidence and visible answer | False admission, fabricated answer, missing evidence, broken tool path |
| Accepted task -> background job -> result recurrence -> adapter | Exact deferred live node above; task/job state and actual delivery receipt/text | Orphaned work, duplicate admission/delivery, wrong audience, unsupported completion |
| Signed DSH interaction -> cognition -> durable decision | Exact internal live node above; signed request, raw judgment and persisted decision | Wrong decision kind, invented evidence or permission |
| RPC/process/SQLite/Mongo recovery | Three explicit diagnostic CLI modes after viability | Authentication, restart, response-loss, uncertain-outcome and resource-lifetime faults |
| Embedded deterministic integration | Existing 25 retained node IDs in the deletion record | Precise handoff, fencing, recurrence, readiness and delivery regressions |
| Entire real LLM test surface | Before/after collection identities and preserved definitions | Accidental deletion, deselection or broken imports |

For each actual production fix, record the exact changed files and failed/passed
probe in the execution evidence. The runtime failure determines the required
evidence; the number of source files does not determine the number of tests.

## Execution and final review

The implementation owner owns the stated code/test/docs surface, isolated
runtime execution, diagnosis, fixes and evidence. Required skills are
probe-first-engineering, development-plan, local-llm-architecture, py-style,
test-style-and-execution, and character-test for live behavior; apply CJK and
UI skills when their actual editing boundaries are involved. The capability
floor is senior async/process/storage and LLM integration engineering.

An independent reviewer has read-only authority over the final diff and runtime
evidence, with the same architectural capability plus character-grounding
review. It reports concrete defects and per-case behavioral acceptance and
cannot remediate its own findings. Resolve the executor at handoff. Review
follows runnable model evidence and does not delay the first real-model probe.

Closure requires the requested reductions and real LLM preservation, successful
foreground/deferred/internal behavior on the final relevant code, observed
recovery and cleanup, and independent acceptance with material defects fixed.
Record actual model calls, usage, elapsed time, raw/parsed outputs, provenance,
visible text and cleanup. Keep failed attempts visible. Archive this plan only
when those outcomes exist; a passing test count cannot close it.

## Progress

- [x] Inventory the whole test surface and preserve a pre-cut/live baseline.
- [ ] Apply ownership/dedicated-suite removals and the embedded integration cut.
- [ ] Verify preserved live collection and surviving imports.
- [ ] Supersede both former active DSH plans and update current guidance.
- [ ] Obtain the first actual foreground model result and fix demonstrated faults.
- [ ] Establish deferred completion and internal judgment with inspected live runs.
- [ ] Inspect recovery, perform justified hardening, and repeat affected probes.
- [ ] Complete independent behavioral/code acceptance and archive this plan.
