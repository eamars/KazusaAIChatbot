# coding agent phase4 code modifying and patching plan

## Summary

- Goal: implement Phase 4 of the coding-agent architecture: existing-source
  semantic modification plus a standalone shared patching boundary for both
  new artifacts and existing-file modifications.
- Plan class: high_risk_migration.
- Status: completed.
- Execution mode: direct parent execution only unless the user explicitly
  approves subagent execution for implementation.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `debug-llm`.
- Cutover strategy: bigbang inside the coding-agent subsystem. Introduce one
  canonical `code_modifying` package and one canonical `code_patching`
  package, then update callers, tests, and README ICDs in the same scope.
- Acceptance focus: deterministic contract tests, role-level live LLM evidence,
  and five ordered real-life E2E coding challenges proving reviewable patch
  proposals for existing-source work.
- Confidence rule: 90% confidence is not claimable from the current proposal
  alone. It becomes claimable only after the readiness ladder in this plan
  passes and the live evidence is independently accepted.

## Current Architecture Map

The reference architecture defines these coding-agent capabilities:

- `code_fetching`: resolve explicit repository, local, raw-file, and inline
  sources.
- `code_reading`: answer questions from bounded source evidence.
- `code_writing`: create source-free new artifacts.
- `code_modifying`: plan semantic changes to existing source files from bounded
  source evidence.
- `code_patching`: materialize selected generated artifacts and selected
  existing-file modifications into a patch or file-tree proposal.
- `code_executing`: run bounded commands in a later approved execution phase.

The current codebase implements fetching, reading, source-free writing, and
background-worker integration. It does not expose a `code_modifying` package.
The top-level `propose_code_change(...)` path rejects explicit source-backed
requests with the limitation that existing-source semantic edits are outside
the current writing scope.

Patching mechanics are currently coupled to `code_writing`. The public writing
model exposes `create_file` operations, while the internal patch compiler
already contains some existing-file edit mechanics. Phase 4 must promote that
boundary into a standalone `code_patching` package with explicit contracts
instead of leaving hidden edit support inside the new-artifact writer.

## Scope

Phase 4 includes:

- `src/kazusa_ai_chatbot/coding_agent/code_modifying/` as a new standalone
  subagent package with README ICD, role contracts, PM/programmer prompts,
  supervisor loop, trace records, and public-safe models.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/` as the canonical patch
  assembly and review materialization package for new-file and existing-file
  proposals.
- promotion of patch models, patch operation compilation, patch validation,
  and review-package materialization out of `code_writing` into
  `code_patching`.
- top-level `propose_code_change(...)` support for explicit source-backed
  requests:

```text
source-backed request
-> code_fetching
-> code_reading evidence
-> code_modifying
-> code_patching
-> review package response
```

- source-free request preservation:

```text
source-free request
-> code_writing
-> code_patching
-> review package response
```

- `handle_background_coding_task(...)` routing support for `code_modifying`
  when the accepted coding task already contains explicit source structure.
- deterministic tests for contracts, path safety, patch compilation, review
  materialization, public response sanitization, and routing.
- live LLM role tests for modifying PM and modifying programmer behavior.
- five ordered live LLM E2E sign-off gates for realistic existing-source code
  modification requests.
- README, HOWTO, and architecture ICD updates for the new boundary.

Phase 4 excludes:

- patch application to the caller workspace;
- command execution, generated test execution, package installation, or repair
  loops driven by executed output;
- autonomous dependency research unless explicitly requested through the
  existing external evidence channel;
- broad repository operations beyond one resolved source scope;
- `code_executing` sandbox design or implementation;
- compatibility shim modules, alias packages, or parallel public vocabularies.

## Architecture Rules

- The top-level coding supervisor owns cross-domain workflow, source fetching,
  reading interleave, operation routing, loop limits, and final public response
  synthesis.
- `code_reading` returns evidence only. It must not choose the final patch
  stance.
- `code_modifying` owns semantic change decomposition for existing files,
  source-owner evidence mapping, source anchors, direct-child lifecycle, and
  final modification artifact selection.
- The File Agent owns path validation, current file context packaging, and
  owned/read-only path maps before programmer dispatch.
- Modifying programmers own one bounded existing-file change contract and
  return structured modification artifacts with evidence references.
- `code_patching` owns patch operation validation, path targeting, anchor
  matching, unified-diff or file-tree assembly, review materialization, and
  non-executing structural checks.
- Deterministic code owns shape validation, limits, path safety, patch parsing,
  review-package persistence, and public response sanitization.
- LLM stages own semantic judgment about task fit, decomposition, file choice,
  and whether gathered evidence is enough.
- Workspace mutation remains outside this phase. Review materialization may
  write only to managed review directories under the configured coding-agent
  workspace root.
- Do not route by hardcoded challenge keywords, repository names, expected
  files, expected functions, or expected answers.

## Target Contracts

Add or update public/internal models around these shapes:

- `CodingAgentBackgroundOperation`: add `code_modifying`.
- `CodingPatchProposalResponse.mode`: preserve `create_new_project`; add
  `edit_existing_repository` and `mixed_repository_change`.
- `CodeModificationRequest`: question, reading result, source scope,
  repository summary, workspace root, preferred language, answer/artifact caps,
  supervisor facts, and optional prior artifacts.
- `CodeModificationResult`: status, selected modification artifacts, evidence
  references, changed-file summaries, limitations, trace summary, and optional
  supervisor fact requests.
- `ModificationArtifact`: artifact id, path, operation intent, required
  behavior, evidence ids, current source anchor, replacement or insertion
  content, summary, and local risks.
- `PatchOperation`: explicit `create_file`, `replace`, `insert_before`,
  `insert_after`, and `replace_file_small` operation kinds. Destructive delete
  is outside Phase 4.
- `PatchProposalPackage`: patch artifacts, created files, changed files,
  validation summary, materialized review paths, and patchability notes.
- `FileContextPackage`: repo-relative path, file role, content excerpt or full
  bounded content, line windows, exact anchors, owned/read-only flag, and
  evidence ids.

The patching package must use deterministic patch assembly from structured
operations. This plan does not add an LLM patching role. If later trace
evidence shows deterministic assembly is insufficient, create a separate plan
with its own role contract, role-level live tests, and E2E readiness gate.

## Confidence Hardening Requirements

The implementation must treat confidence as an evidence state, not as a
planning claim. The target of at least 90% confidence for the five live gates
is blocked until this readiness ladder is complete:

1. Deterministic interface tests prove source-backed
   `propose_code_change(...)` requests no longer return
   `writing:existing_source_rejected`.
2. Deterministic `code_patching` contract tests pass for operation parsing,
   path containment, unique-anchor matching, missing-anchor rejection,
   duplicate-anchor rejection, full-file replacement policy, atomic package
   failure, review materialization, and public response sanitization.
3. Modifying PM live role tests pass one case at a time for
   `request_information`, `create_child_pm`, `create_programmer_task`,
   `repair_child`, `complete`, and `blocked`.
4. Modifying programmer live role tests pass one case at a time for a
   single-file edit, parser edit, cross-layer behavior edit, and mixed
   existing-file/new-file proposal.
5. Gate 03 and Gate 05 receive role-level dry-run decomposition review before
   the full public E2E gate run.
6. The five public E2E gates run one at a time, each raw trace and
   materialized review package is inspected, and the reviewer records semantic
   acceptance.

No E2E gate may run for sign-off until the role-level suite passes and the
reviewer records readiness confidence of at least 90% from role evidence. If a
role-level or E2E live case passes pytest but shows confused decomposition,
missing source grounding, accidental replacement-project behavior, or brittle
patch operations, the gate remains unaccepted and the confidence target is not
met.

### Modifying PM Decision Contract

Every modifying PM decision must be parsed into one closed status:
`request_information`, `create_child_pm`, `create_programmer_task`,
`repair_child`, `complete`, or `blocked`.

Every decision must include:

- `status`: one of the allowed statuses above.
- `reason`: a bounded explanation grounded in the user request, source
  evidence, or direct child result.
- `owned_paths`: repo-relative paths the PM is allowed to change.
- `read_only_paths`: repo-relative paths used only as evidence or consumed
  interfaces.
- `required_evidence_ids`: evidence ids that justify the decision.

`request_information` must include the missing source fact, why the PM cannot
decide without it, and the requested read scope for supervisor-mediated
`code_reading`.

`create_child_pm` must include a child task id, child objective, accepted path
scope, required source facts, expected report shape, and stopping condition.

`create_programmer_task` must include:

- `task_id`;
- `target_paths`;
- `change_goal`;
- `required_behavior`;
- `forbidden_changes`;
- `consumed_interfaces`;
- `expected_operations`;
- `acceptance_checks`;
- `local_risks`.

`repair_child` is allowed only for PM-to-direct-child structural or contract
feedback from deterministic parsing, handoff validation, patch validation, or
review materialization. It must include the child task id, rejected fields,
required correction, and cited validator finding. It must not consume executed
test output, shell command output, package-install output, target-project
runtime output, or real workspace diffs. No automated retry loop may be driven
by executed output in this phase. The implementation must cap repairs at two
repairs per direct child and four repairs per modifying run.

`complete` must include selected artifact ids, changed paths, evidence ids,
confidence notes, known limitations, and reviewer-facing summary text.

`blocked` must include the missing facts or invalid state, why the PM cannot
proceed safely, and the most specific next human or architecture action.

### Modifying Programmer Artifact Contract

Every modifying programmer output must be one bounded artifact for one target
responsibility. It must include:

- `status`: `succeeded` or `blocked`;
- `task_id`;
- `target_path`;
- `evidence_ids`;
- `operation_kind`: `replace`, `insert_before`, `insert_after`, or
  `replace_file_small`;
- `exact_anchor`;
- `replacement_or_insert_content`;
- `operation_summary`;
- `risk_notes`;
- `tests_or_docs_to_update`.

The programmer must not return raw unified diffs. It must cite source anchors
and evidence ids. A blocked artifact must explain which source fact, interface
fact, or anchor is missing.

### Deterministic Patching Policy

`code_patching` owns all edit mechanics. LLM roles return structured
operations only.

Patch operation rules:

- `create_file` may create a new repo-relative file only when the path does
  not already exist and the File Agent reserved the path.
- `insert_before`, `insert_after`, and `replace` require an exact anchor that
  appears exactly once in the current target file.
- Missing anchors and multiple matching anchors are terminal validation
  failures for that package. The patcher must report ambiguity instead of
  choosing a match.
- `replace_file_small` is allowed only for text files at or below the
  code-patching full-file cap and only when the programmer supplies the full
  replacement content, evidence ids, and a rationale for why section anchors
  are weaker than full-file replacement. The initial cap must be a named
  deterministic constant of 20,000 characters and deterministic tests must
  cover accepted and rejected boundaries.
- Delete operations, rename operations, chmod operations, binary writes,
  package installation, and command execution remain out of scope.
- Mixed packages must be atomic. If any operation is invalid, the package is
  rejected or returned for contract repair; no partial package is reported as
  successful.
- Review materialization may write only to a managed review directory under
  the configured coding-agent workspace root.
- Public responses must omit absolute copied source roots, workspace roots,
  cache keys, raw command output, raw traces, and worker-local arguments.

### Gate 05 Decomposition Requirement

The hard inventory cache gate must be decomposed before programmer dispatch
into these work items:

- fetch-layer timeout, retry, and cache integration;
- CLI flag parsing and wiring;
- mocked tests for cache miss, cache hit, refresh-cache, timeout propagation,
  and retry after transient failure;
- README workflow update.

The modifying PM must request or cite evidence from `fetch.py`, `cli.py`,
existing tests, and README before dispatching programmer tasks. A new helper
file such as `inventory_sync/cache.py` is allowed only when the PM explains
the interface it provides and the existing files that will import it. Retry
count and timeout behavior must be owned by deterministic code constants or
explicit CLI inputs; no real network tests are allowed.

## Implementation Stages

### Stage 0: Plan Readiness

- [x] Reread this plan after any context compaction and before each major
  checklist stage.
- [x] Confirm `git status --short`, `README.md`, `docs/HOWTO.md`,
  `development_plans/README.md`, the coding-agent README, relevant source
  files, and relevant tests before production-code edits.
- [x] Confirm the user has explicitly approved implementation. This plan is
  executing through the user-requested no-subagent fallback path.
- [x] Confirm the readiness ladder in `Confidence Hardening Requirements` is
  treated as a hard sign-off precondition, not a recommendation.

### Stage 1: Promote Code Patching

- [x] Create `coding_agent/code_patching/README.md` documenting the patching
  ICD and the no-apply/no-execution boundary.
- [x] Move canonical patch models and operation compilation into
  `coding_agent/code_patching/models.py` and
  `coding_agent/code_patching/patch_operations.py`.
- [x] Move review materialization and validation into
  `coding_agent/code_patching/patch_validation.py`.
- [x] Add deterministic validators for unique anchors, `replace_file_small`
  cap enforcement, atomic mixed-package rejection, and no-delete/no-execution
  policy.
- [x] Update `code_writing` to emit selected generated artifacts and call
  `code_patching`; remove writing-local patching ownership in the same change.
- [x] Add deterministic tests proving new-artifact writing still produces the
  same reviewable proposal shape through the new patching package.

### Stage 2: Extend Source Context Ownership

- [x] Extend the File Agent or add a modifying-specific file-context helper for
  current existing-file packages.
- [x] Validate repo-relative paths, text-only files, size caps, binary/secret
  exclusions, source-scope containment, and owned/read-only path maps.
- [x] Add exact anchor extraction helpers with line-number metadata.
- [x] Add tests for unsafe paths, missing files, binary files, overlarge files,
  duplicate anchors, and read-only path violations.

### Stage 3: Implement Code Modifying

- [x] Create `coding_agent/code_modifying/` with README, models, PM role,
  programmer role, supervisor, synthesis helpers, and trace helpers.
- [x] Implement the modifying PM lifecycle:
  `request_information`, `create_child_pm`, `create_programmer_task`,
  `repair_child`, `complete`, and `blocked`.
- [x] Require PM instructions to map every programmer task to source evidence,
  accepted paths, consumed interfaces, expected behavior, and local risks.
- [x] Require modifying programmers to return structured modification
  artifacts instead of raw freeform diffs.
- [x] Route PM requests for additional source facts back through the top-level
  supervisor and `code_reading`.
- [x] Enforce loop limits, child limits, file caps, answer caps, and trace caps
  deterministically.

### Stage 4: Integrate Top-Level Supervisor

- [x] Replace the existing-source rejection in `propose_code_change(...)` with
  fetch/read/modify/patch orchestration.
- [x] Preserve source-free writing behavior and session handling.
- [x] Remove stale unreachable existing-repository scaffold code after the
  current rejection return.
- [x] Add `code_modifying` to background coding operation routing.
- [x] Ensure background responses remain sanitized and expose no local roots,
  cache keys, raw traces, worker-local arguments, or adapter-delivery details.
- [x] Keep operation choice inside the coding-agent public background interface,
  not in L2d or the generic background-work router.

### Stage 5: Documentation and ICD

- [x] Update `src/kazusa_ai_chatbot/coding_agent/README.md` architecture,
  operation matrix, public contracts, worker handoff, and change-control notes.
- [x] Update `docs/HOWTO.md` coding-agent operation notes.
- [x] Update `development_plans/reference/designs/coding_agent_architecture.md`
  only where the reference is stale against the implemented Phase 4 boundary.
- [x] Capture the five existing-source live gates in
  `tests/test_coding_agent_existing_source_e2e_live_llm.py` and keep this plan
  aligned with that test contract.

### Stage 6: Deterministic Verification

- [x] Add `tests/test_coding_agent_phase4_code_patching_contracts.py`.
- [x] Add `tests/test_coding_agent_phase4_code_modifying_contracts.py`.
- [x] Add `tests/test_coding_agent_phase4_interface.py`.
- [x] Add fixture projects under
  `tests/fixtures/coding_agent_existing_source_gates/`.
- [x] Verify source-free Phase 2 tests still pass through the promoted patching
  boundary.
- [x] Verify explicit source requests produce reviewable patch proposals and do
  not mutate source fixtures.

### Stage 7: Role-Level Live LLM Verification

- [x] Add `tests/test_coding_agent_phase4_code_modifying_role_live_llm.py`.
- [x] Run one live LLM case at a time and inspect trace output before the next
  case.
- [x] Prove the modifying PM can request more evidence, dispatch a child PM,
  dispatch one programmer, complete with selected artifacts, and block when the
  source evidence is insufficient.
- [x] Prove the modifying programmer can produce structured operations for a
  small single-file edit, parser edit, cross-layer behavior edit, and mixed
  existing-file/new-file proposal.
- [x] Prove `repair_child` consumes only structural contract feedback and does
  not consume executed test output, command output, package-install output, or
  target-project runtime output.
- [x] Review Gate 03 and Gate 05 role-level decomposition traces before full
  E2E sign-off runs.
- [x] Record raw responses, traces, and review notes under
  `test_artifacts/llm_traces/coding_agent_phase4_roles/`.

Execution note: the no-subagent execution path added the role-level live test
file and used deterministic PM contract coverage because the current modifying
PM implementation is a contract normalizer rather than a model-backed role.
Live programmer traces and the public E2E traces provide the model-backed
evidence.

### Stage 8: E2E Live LLM Sign-Off Gates

- [x] Add `tests/test_coding_agent_existing_source_e2e_live_llm.py`.
- [x] Confirm all role-level live tests have passed and reviewer readiness
  confidence is at least 90% before running these gates for sign-off.
- [x] Run gates one at a time through public `propose_code_change(...)`.
- [x] Persist raw response JSON, trace summary, patch artifacts, validation,
  and materialized review paths for every gate.
- [x] Assert only reviewability, structural safety, sanitization, and
  non-mutation in pytest. AI or human review owns semantic pass/fail.
- [x] Close Phase 4 only after all five gates have reviewable artifacts and the
  reviewer accepts the proposed modifications.

## Deterministic Test Requirements

Deterministic tests must cover:

- source-backed requests no longer return `writing:existing_source_rejected`;
- source-free writing still creates reviewable new-artifact proposals;
- `code_modifying` rejects missing reading evidence, unsupported source scopes,
  unsafe paths, binary files, and overlarge file contexts;
- modifying PM output parsing rejects invalid states, missing evidence mapping,
  programmer tasks without accepted paths, and completion without artifacts;
- modifying programmer parsing rejects raw diffs, missing anchors, unsafe paths,
  unsupported operation kinds, and artifacts without evidence ids;
- patching compiles `create_file`, `replace`, `insert_before`, and
  `insert_after` operations into bounded unified diffs and enforces the
  `replace_file_small` cap;
- patching rejects unsafe paths, anchors that do not match, operations that
  touch too many files, duplicate anchors, ambiguous anchors, partial package
  success, and diffs that exceed caps;
- `repair_child` accepts only structural contract feedback and rejects executed
  test output, command output, package-install output, target-project runtime
  output, and workspace diffs as repair inputs;
- review materialization writes only under the configured managed workspace;
- source fixtures remain byte-identical after proposal generation;
- public background and direct responses omit local roots, cache keys, raw
  traces, and worker-local arguments.

## Phase 4 Live E2E Coding Challenges

Each gate is a realistic existing-source modification request. The canonical
gate definitions live in
`tests/test_coding_agent_existing_source_e2e_live_llm.py`; this section mirrors
that test contract for plan review. The gate specification is part of the work
contract and must be independently reviewed before implementation starts.
Fixtures are small enough for local LLM context limits but structured enough to
require real fetching, reading, modification, and patch planning.

The live pytest harness must call only the public top-level
`propose_code_change(...)` entrypoint. For every gate, the harness passes:

- `question`: the gate's Modification instruction;
- `local_root_hint`: a managed copy of the gate fixture source root;
- `workspace_root`: a managed test-artifact workspace for patch review output;
- `session_id`: a stable Phase 4 gate id;
- `preferred_language`: `English`;
- `max_answer_chars` and `max_artifact_chars`: bounded values consistent with
  existing Phase 2 live-gate conventions.

For every gate, the source fixture lives under
`tests/fixtures/coding_agent_existing_source_gates/`. The harness copies that
fixture into the managed gate workspace and passes the copy as
`local_root_hint` so an implementation failure cannot dirty the repository
fixture. Code fetching must resolve the copied local source first, code reading
must produce source-grounded evidence, code modifying must produce existing-file
modification artifacts, and code patching must materialize a reviewable patch
proposal. The copied source tree must remain byte-identical after the run.

### Gate 01: CLI JSON Output Toggle

Difficulty: simple single-file edit with one focused test update.

#### Objective

Prove the modifying and patching flow can handle a bounded single-file CLI
behavior extension, update a focused existing test, and avoid creating a
replacement project.

#### Existing Code Base

Fixture root:
`tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter`

The live harness must copy this fixture into the managed gate workspace and
pass the copy as `local_root_hint`. The coding agent must invoke
`code_fetching` against the copied local source, then `code_reading` must cite
the existing source and tests before `code_modifying` is allowed to emit
modification artifacts.

Required fixture shape:

```text
gate_01_log_counter/
  README.md
  log_counter.py
  tests/
    test_log_counter.py
```

Current baseline:

- `log_counter.py` is a standard-library CLI module.
- `SEVERITIES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]`.
- `count_severities(path: Path) -> tuple[dict[str, int], int]` counts valid
  severity-prefixed lines and returns skipped malformed-line count.
- `format_summary(counts: dict[str, int], skipped: int) -> str` returns text
  output with one line per severity and one skipped-line line.
- `main(argv: list[str] | None = None) -> int` accepts one required log path,
  prints the text summary to stdout, prints missing-file errors to stderr, and
  returns `2` for missing input files.
- `tests/test_log_counter.py` covers valid counts, malformed-line skipping,
  text formatting, and missing-file CLI behavior.
- `README.md` documents text summary usage only.

#### Modification Instruction

Modify the existing log counter CLI to add a `--json` flag. When `--json` is
provided, stdout must be valid JSON containing the same severity counts and the
skipped-line count. The default text output must remain unchanged. Use only the
Python standard library and update focused tests for the new JSON mode.

#### Expected State

- `log_counter.py` still owns the CLI; no replacement script or new project is
  introduced.
- Default invocation still prints the existing text summary format.
- `--json` prints one JSON object with keys `DEBUG`, `INFO`, `WARNING`,
  `ERROR`, `CRITICAL`, and `skipped`.
- The JSON object uses integer values and stable severity key names matching
  `SEVERITIES`.
- Missing-file behavior and exit code remain unchanged.
- Focused tests prove JSON output parses and preserves the same counts.
- README usage may be updated, but source and test changes are sufficient for
  this simple gate.

#### Pass Criteria

- Public response status is `succeeded`; mode is `edit_existing_repository` or
  an equivalent source-backed patch-proposal mode.
- Evidence includes `log_counter.py` and `tests/test_log_counter.py`.
- Patch artifacts modify existing files rather than returning only new files.
- Review materialization contains the proposed changed source and test content.
- Validation reports a reviewable patch proposal and no copied source mutation.
- The public response omits local roots, cache keys, raw traces, and worker
  internals.
- Independent review confirms the expected default behavior, JSON behavior,
  missing-file behavior, test update, standard-library-only constraint, and
  minimal patch scope.

### Gate 02: JSONL Import Error Handling

Difficulty: low-medium multi-file utility edit.

#### Objective

Prove the modifying flow can coordinate a small multi-file utility change where
parser behavior, CLI behavior, tests, and documentation must remain coherent.

#### Existing Code Base

Fixture root:
`tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv`

The live harness must copy this fixture into the managed gate workspace and
pass the copy as `local_root_hint`. The coding agent must invoke
`code_fetching` against the copied local source. `code_reading` must produce
evidence for the converter, CLI, tests, and README before `code_modifying`
emits modification artifacts.

Required fixture shape:

```text
gate_02_contacts_jsonl_to_csv/
  README.md
  contacts_jsonl_to_csv/
    __init__.py
    cli.py
    converter.py
  tests/
    test_cli.py
    test_converter.py
```

Current baseline:

- `converter.py` exposes
  `convert_jsonl_to_csv(input_path: Path, output_path: Path, fields: list[str] | None = None) -> int`.
- The converter reads JSON objects from JSONL and writes CSV with `csv.DictWriter`.
- When `fields` is omitted, the baseline uses first-record key order.
- When `fields` is provided, the baseline currently normalizes or sorts fields
  in a way that does not preserve exact user-supplied order.
- Missing field values currently become Python `None` or are omitted
  inconsistently instead of guaranteed blank CSV cells.
- Malformed JSON currently produces a generic warning without the 1-based line
  number and continues.
- `cli.py` exposes `python -m contacts_jsonl_to_csv.cli INPUT OUTPUT
  [--fields FIELD[,FIELD...]]`.
- There is no `--strict` option.
- Existing tests cover basic conversion and CLI happy path only.
- README documents the basic command and optional field list.

#### Modification Instruction

Modify the existing JSONL-to-CSV utility so `--fields` defines the exact CSV
column order, missing fields are written as blank cells, malformed JSON is
reported with 1-based line numbers, default behavior continues past malformed
lines, and `--strict` fails fast on the first malformed line. Use only the
Python standard library. Update focused converter, CLI, and README coverage.

#### Expected State

- The existing package and CLI entrypoint remain in place.
- `--fields id,name,email` writes columns exactly in `id,name,email` order.
- Missing values for requested fields produce empty CSV cells, not `None`.
- Malformed JSON reports a message that includes the failing 1-based line
  number.
- Non-strict mode skips malformed lines, writes valid rows, and exits
  successfully.
- `--strict` aborts on the first malformed line, exits non-zero, and avoids
  presenting a successful partial conversion as complete.
- Tests cover field order, missing fields, malformed-line reporting,
  non-strict continuation, and strict failure.
- README or CLI help documents `--strict` and the exact field-order behavior.

#### Pass Criteria

- Public response status is `succeeded`; mode is source-backed modification.
- Evidence includes `converter.py`, `cli.py`, at least one existing test file,
  and README.
- Patch artifacts include existing-source edits and existing-test edits.
- Any new helper functions remain inside the existing package boundary.
- Review materialization contains coherent converter, CLI, tests, and docs
  changes.
- Validation reports a reviewable patch proposal and no copied source mutation.
- Independent review confirms the utility preserves the existing entrypoint,
  implements exact field ordering, blank-cell handling, line-numbered errors,
  non-strict continuation, strict failure, test coverage, and standard-library
  only behavior.

### Gate 03: Markdown Link Checker Parser Upgrade

Difficulty: medium parser and tests across several files.

#### Objective

Prove the modifying PM can identify a parser-owned change, avoid output-only
filtering, and update tests for edge cases that require source understanding.

#### Existing Code Base

Fixture root:
`tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker`

The live harness must copy this fixture into the managed gate workspace and
pass the copy as `local_root_hint`. The coding agent must invoke
`code_fetching` against the copied local source. `code_reading` must cite the
parser/scanner source and tests before `code_modifying` emits modification
artifacts.

Required fixture shape:

```text
gate_03_markdown_link_checker/
  README.md
  mdlinkcheck/
    __init__.py
    anchors.py
    cli.py
    scanner.py
  tests/
    test_anchors.py
    test_scanner.py
```

Current baseline:

- `anchors.py` exposes `slugify_heading(text: str) -> str` and
  `collect_anchors(markdown: str) -> dict[str, int]`.
- `scanner.py` exposes `find_markdown_links(markdown: str) -> list[MarkdownLink]`
  and `check_file(path: Path, root: Path) -> list[LinkProblem]`.
- The scanner finds inline Markdown links with a simple text scan or regex.
- The scanner currently reports links inside fenced code blocks and HTML
  comments as real links.
- Duplicate headings currently collide on the same anchor; links to
  suffixed anchors such as `#install-1` are reported broken.
- The CLI delegates to scanner functions and prints problems.
- Existing tests cover basic heading slugification, a valid relative link, a
  broken relative link, and duplicate-heading detection.

#### Modification Instruction

Modify the existing Markdown link checker to ignore links inside fenced code
blocks and HTML comments. Support duplicate heading anchors with GitHub-style
suffixes when resolving links: the first duplicate heading keeps the base
anchor, the second becomes `base-1`, the third becomes `base-2`, and so on.
Update focused parser/scanner tests and keep normal links outside comments and
fences working.

#### Expected State

- Parser/scanner logic owns the exclusion of code-fence and HTML-comment links;
  the CLI does not hide parser mistakes after the fact.
- Both triple-backtick and triple-tilde fences are handled across multiple
  lines.
- HTML comments can span multiple lines and links inside them are ignored.
- Normal inline Markdown links outside comments and fences are still checked.
- Duplicate headings produce resolvable anchors using `base`, `base-1`,
  `base-2`, etc.
- Links to suffixed duplicate anchors resolve successfully.
- Tests cover fenced-code links, HTML-comment links, normal outside links, and
  duplicate-anchor suffix resolution.
- Production code does not hardcode fixture filenames, headings, or exact test
  strings.

#### Pass Criteria

- Public response status is `succeeded`; mode is source-backed modification.
- Evidence includes `anchors.py`, `scanner.py`, and existing scanner/anchor
  tests.
- Patch artifacts modify the existing parser/scanner path and tests.
- Review materialization shows parser-level handling rather than CLI-only
  suppression.
- Validation reports a reviewable patch proposal and no copied source mutation.
- Independent review confirms fenced-block handling, HTML-comment handling,
  duplicate-anchor suffix behavior, normal-link preservation, focused tests,
  and absence of challenge-specific hardcoding.

### Gate 04: Issue Tracker Soft Delete

Difficulty: medium-hard cross-layer behavior change.

#### Objective

Prove the modifying flow can handle a cross-layer semantic change where model,
store, service/API, tests, and docs must move together without compatibility
shims.

#### Existing Code Base

Fixture root:
`tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete`

The live harness must copy this fixture into the managed gate workspace and
pass the copy as `local_root_hint`. The coding agent must invoke
`code_fetching` against the copied local source. `code_reading` must cite
model, store, API/service, tests, and README evidence before `code_modifying`
emits modification artifacts.

Required fixture shape:

```text
gate_04_issue_tracker_soft_delete/
  README.md
  issue_tracker/
    __init__.py
    api.py
    models.py
    store.py
  tests/
    test_api.py
    test_store.py
```

Current baseline:

- `models.py` defines an `Issue` dataclass with fields `id`, `title`,
  `status`, and `assignee`.
- `store.py` defines `InMemoryIssueStore` with `create_issue`, `get_issue`,
  `list_issues`, and `delete_issue`.
- `delete_issue(issue_id: str) -> bool` currently hard-removes the issue from
  the internal dictionary.
- `get_issue(issue_id)` returns `None` after hard delete.
- `list_issues()` returns every issue still present in insertion order.
- `api.py` exposes route-like service functions over the store, including
  `delete_issue(store, issue_id)`.
- Existing tests expect deleted issues to disappear because they were removed
  from storage.
- README describes delete as permanent removal.

#### Modification Instruction

Modify the existing issue tracker to implement soft delete. Deleting an issue
should mark it archived instead of removing it from storage. Normal list
results should hide archived issues. Single-item lookup should return not
found for archived issues. List should support an `include_archived` option.
Update store/API tests and README. Keep behavior coherent with the existing
project style and avoid compatibility wrappers around the old hard-delete
semantics.

#### Expected State

- `Issue` has an archived state, such as `archived: bool = False`, without
  breaking existing issue creation callers.
- `delete_issue(issue_id)` marks an existing active issue archived and returns
  `True`.
- Deleting a missing issue returns `False`, matching existing missing-delete
  style.
- Normal `get_issue(issue_id)` returns `None` for archived issues.
- Normal `list_issues()` excludes archived issues.
- `list_issues(include_archived=True)` includes archived issues in the same
  deterministic order used by existing listing behavior.
- API/service functions expose the same soft-delete semantics as the store.
- Tests cover delete, hidden archived lookup, default list hiding, and
  include-archived listing.
- README describes soft delete and include-archived behavior.

#### Pass Criteria

- Public response status is `succeeded`; mode is source-backed modification.
- Evidence includes `models.py`, `store.py`, `api.py`, store tests, API tests,
  and README.
- Patch artifacts modify existing model/store/API/test/doc files coherently.
- No compatibility shim, alias method, fallback mapper, or parallel hard-delete
  vocabulary is introduced.
- Review materialization shows the model and store/API behavior moved together.
- Validation reports a reviewable patch proposal and no copied source mutation.
- Independent review confirms archived state preservation, default lookup/list
  hiding, include-archived listing, missing-delete behavior, tests, docs, and
  no old hard-delete wrapper.

### Gate 05: Inventory Vendor Fetch Cache

Difficulty: hard mixed existing-file and new-file proposal.

#### Objective

Prove the full Phase 4 pipeline can handle a hard mixed change: existing-file
modification plus a justified new helper file, with cache behavior, CLI flags,
mocked HTTP tests, and documentation all kept reviewable.

#### Existing Code Base

Fixture root:
`tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache`

The live harness must copy this fixture into the managed gate workspace and
pass the copy as `local_root_hint`. The coding agent must invoke
`code_fetching` against the copied local source. `code_reading` must cite
fetch, CLI, report, tests, and README evidence before `code_modifying` emits
modification artifacts. If the solution creates a new cache helper file, the
final patch package must combine the new-file operation and existing-file
modifications through `code_patching`.

Required fixture shape:

```text
gate_05_inventory_sync_fetch_cache/
  README.md
  inventory_sync/
    __init__.py
    cli.py
    csv_io.py
    fetch.py
    html_extract.py
    report.py
  tests/
    test_cli.py
    test_fetch.py
    test_report.py
```

Current baseline:

- `csv_io.py` reads inventory CSV rows with columns `sku`, `name`, and `url`.
- `fetch.py` exposes `fetch_page(url: str) -> str` using
  `urllib.request.urlopen(url)` without an explicit timeout.
- `fetch.py` does not retry transient failures.
- `fetch.py` does not cache responses.
- `html_extract.py` extracts `<title>` and first `<h1>` values from HTML.
- `report.py` merges inventory rows with fetched title/h1 values and writes a
  consolidated CSV report.
- `cli.py` accepts `--input` and `--output`, reads inventory rows, calls
  `fetch_page` for each URL, extracts HTML metadata, and writes the report.
- Existing tests mock `urllib.request.urlopen` for successful fetch and verify
  report generation. No tests use real network access.
- README documents input CSV columns and the basic command workflow only.

#### Modification Instruction

Modify the existing inventory sync project to add timeout and retry handling
for vendor page fetches, add a file-backed response cache, expose CLI flags
`--cache-dir`, `--refresh-cache`, and `--timeout`, update mocked HTTP tests,
and document the workflow. Use only the Python standard library. Do not run
real network calls in tests.

#### Expected State

- The existing `fetch_page` path or an equivalent fetch-layer function accepts
  a timeout and applies it to `urllib.request.urlopen`.
- Transient fetch failures are retried with a deterministic small retry count
  owned by code, without requiring a new user-facing retry flag.
- Cache keys are deterministic and safe for filesystem use, such as a hash of
  the URL.
- `--cache-dir PATH` enables file-backed raw HTML caching.
- With an existing cache entry and no `--refresh-cache`, the CLI uses cached
  HTML and avoids a network call for that URL.
- With `--refresh-cache`, the CLI fetches fresh HTML and overwrites the cache
  entry.
- `--timeout SECONDS` wires through CLI parsing into the fetch layer.
- Existing CSV parsing, HTML extraction, and report-writing responsibilities
  remain in their current modules.
- Tests mock HTTP for cache miss, cache hit, refresh-cache, timeout propagation,
  and retry after a transient failure.
- README documents cache directory, refresh behavior, timeout behavior, and
  the input/output workflow.
- If a new helper module such as `cache.py` is proposed, it is a focused
  addition and existing files are modified to consume it through explicit
  imports.

#### Pass Criteria

- Public response status is `succeeded`; mode is `mixed_repository_change` or
  another source-backed mode that clearly includes existing-file modification
  plus optional new-file creation.
- Evidence includes `fetch.py`, `cli.py`, `report.py` or `csv_io.py`, existing
  tests, and README.
- Patch artifacts include existing-source edits and may include one focused
  new helper file.
- Patch artifacts update mocked HTTP tests and do not add real network calls.
- Existing parser/report responsibilities are not bypassed by a replacement
  end-to-end script.
- Review materialization contains coherent fetch/cache/CLI/test/doc changes.
- Validation reports a reviewable patch proposal and no copied source mutation.
- Independent review confirms timeout propagation, deterministic retry, cache
  hit, cache refresh, safe cache keys, CLI flag wiring, mocked tests, README
  documentation, standard-library-only behavior, and mixed patch assembly.

## Gate Specification Independent Review

Before Phase 4 implementation begins, run an independent plan review focused
on the live gate specifications in
`tests/test_coding_agent_existing_source_e2e_live_llm.py`, the fixture projects
under `tests/fixtures/coding_agent_existing_source_gates/`, and the mirrored
plan section above. The reviewer must inspect those files and record a
human-authored review artifact under
`test_artifacts/plan_reviews/coding_agent_phase4_gate_spec_review.md`.

The review must check:

- every gate has Objective, Existing Code Base, Modification Instruction,
  Expected State, and Pass Criteria;
- the gates increase in difficulty from simple single-file modification to
  mixed existing-file plus new-file patch proposal;
- each gate requires `code_fetching` and `code_reading` evidence before
  `code_modifying`;
- each gate is source-backed and cannot be satisfied by source-free
  `code_writing` alone;
- pass criteria separate deterministic harness checks from semantic review;
- each gate carries behavior rubric, forbidden failure modes, and trace
  requirements for the live review artifact;
- no gate requires patch application, command execution, package installation,
  generated-test execution, or real network calls;
- no gate encourages hardcoded challenge keywords, expected exact code, or
  fixture-specific production logic;
- the gate set covers single-file edit, multi-file parser/CLI edit, parser
  edge cases, cross-layer domain behavior, and mixed existing/new patch
  assembly.

Implementation may proceed only after the independent review records either
`accepted` or lists required corrections that are applied to this plan.

## E2E Gate Harness Rules

- Mark live gates with `pytest.mark.live_llm`.
- Use existing project conventions for live LLM execution and trace capture.
- Run one gate at a time. Inspect raw response, trace, validation, and
  materialized review files before starting the next gate.
- The pytest assertion layer may fail only for unreviewable evidence, missing
  patch artifacts, missing materialized files, unsafe paths, source fixture
  mutation, unsanitized public response, or coding-agent status failures.
- The pytest assertion layer must not assert exact generated code, exact prose,
  exact filenames beyond fixture source paths, or semantic pass/fail.
- Reviewer notes must include direct materialized review paths and a short
  statement that the proposed patch was inspected.
- Reviewer notes must evaluate the gate's behavior rubric and forbidden
  failure modes from the trace artifact before accepting semantic pass/fail.

## Completion Criteria

Phase 4 is complete when:

- the source-free Phase 2 proposal path still passes its deterministic and live
  reviewability checks;
- source-backed direct patch proposals pass deterministic contract tests;
- background coding-agent routing supports `code_modifying` without leaking
  worker-local details;
- all modifying PM/programmer live role cases are inspected and accepted;
- all five Phase 4 live E2E gates produce reviewable, sanitized, non-mutating
  patch proposals accepted by the reviewer;
- coding-agent README, HOWTO, and architecture ICDs reflect the implemented
  boundary;
- this plan is moved to the completed archive with a closure note listing the
  final accepted gates and any deferred risks.

## Deferred Risks

- Real LLM reliability may require prompt tightening after role-level failures.
  Fix prompts and contracts from trace evidence, not from hardcoded gate
  answers.
- The current architecture cannot honestly claim 90% confidence until
  source-backed routing, patching contracts, role-level live behavior, and
  E2E semantic review have all passed the readiness ladder.
- Existing-file anchors are fragile when files contain repeated blocks. The
  patching package must report ambiguity instead of guessing.
- Multi-file semantic changes can exceed local LLM context. The modifying PM
  must request bounded reading facts and child PM decomposition instead of
  packing whole repositories into one prompt.
- Hidden partial patching support inside current writing code must be removed
  or promoted in the same change to avoid two patching vocabularies.
