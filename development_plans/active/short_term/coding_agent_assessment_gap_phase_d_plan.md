# coding_agent_assessment_gap_phase_d_plan

## Summary

- Goal: Replace the coding agent's fixed repository-exploration and
  proposal/repair pipelines with one bounded, persistent JSON action loop that
  can read, search, edit, run approved verification, delete files, rename
  files, and use a complete persistent repository index without file, wave, or
  report-count exploration caps.
- Plan class: high-risk migration
- Status: in_progress
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`
- Dependencies: Complete Phase B and Phase C first. Phase D consumes Phase B's
  proposal-bound candidate workspace, source-free preflight policy, execution
  planner, repair failure bundle, and revision binding. It consumes Phase C's
  durable continuation context, typed blocker resume targets, operation locks,
  and versioned 30-case benchmark seam.
- Dependency evidence: Phase B is completed at
  `development_plans/archive/completed/short_term/coding_agent_assessment_gap_phase_b_plan.md`.
  Phase C is completed at
  `development_plans/archive/completed/short_term/coding_agent_assessment_gap_phase_c_plan.md`.
  The Phase C benchmark manifest exists at
  `tests/fixtures/coding_agent_benchmark/cases.jsonl`; its three inspected
  smoke artifacts are not a 30-case `pipeline_v1` baseline.
- Overall cutover strategy: Build the new loop and index behind an
  evaluation-only entry point, run the Phase C benchmark one case at a time,
  then perform one production cutover for `read_only`, `propose_patch`, and
  `verify_repair`. Remove the superseded manager/programmer generation paths in
  the same cutover; do not retain a runtime fallback or compatibility adapter.
- Highest-risk areas: stale repository index results, unconstrained looping,
  edit precondition drift, delete/rename review ambiguity, candidate/base
  divergence, and accidentally exposing commands or operational policy to the
  model.
- Acceptance criteria: All three coding objectives use the same strict JSON
  action protocol; every safe text file is discoverable without repository-map
  file caps; candidate edits are searchable immediately; delete and rename are
  first-class reviewable patch operations; run actions remain deterministic
  and policy-bound; the Phase C benchmark clears the cutover gate with no
  safety, privacy, approval, or source-integrity regression.

## Context

The capability assessment identifies a structural limitation rather than one
missing specialist. The current implementation performs repository exploration
through fixed manager/programmer waves, bounded file lists, bounded reports,
and a repository map capped at 120 files. Proposal generation and repair then
restart work through separate `code_writing` and `code_modifying` pipelines.
Those stages can produce useful results, but the model cannot repeatedly choose
the next most useful observation, edit the candidate, run available checks, and
continue from the resulting evidence in one coherent working loop.

The current patch boundary also deliberately excludes delete and rename. That
keeps the existing patcher narrow, but prevents common repository work such as
removing obsolete modules, renaming a public module, or moving a file while
preserving its content and review history.

The repository-scale reference proposes more manager hierarchy and strict
fan-out caps. Phase D supersedes that direction for coding-agent exploration.
Repository scale is handled by a persistent deterministic index and a
single-controller action loop, not by adding another model-management layer.

The target remains a local-LLM architecture. The model receives a compact task,
working state, and bounded observations. Deterministic code owns source scope,
index construction, action validation, candidate mutation, execution policy,
limits, persistence, locks, patch assembly, and approval binding.

Current pipeline-shaped tests are not an architectural authority for this
phase. Phase D defines target-state contracts from the production boundary and
the requested direction. During implementation, existing tests are only a
migration inventory for removal or replacement after the new contracts exist;
they must not force preservation of the old manager/programmer topology.

### Assessment Coverage And Handoff

| Requested capability | Phase D disposition |
|---|---|
| Generic JSON read/search/edit/run loop | Close with one strict action protocol and one durable controller for all coding objectives. |
| Delete/rename | Close with guarded candidate operations, patch representation, review materialization, and apply validation. |
| Persistent repository indexing | Close with versioned workspace-local SQLite/FTS snapshots and a live candidate overlay. |
| Removal of exploration caps | Remove file, repository-map, wave, programmer, and report-count caps; retain only run-level resource and no-progress budgets. |
| Current test alignment | Treat existing tests as non-authoritative and create replacement target-state verification. |
| Broader tools, package installation, publishing, or arbitrary shell | Defer; these are outside Phase D. |

## Mandatory Skills

- `development-plan`: load before changing or executing this plan.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before creating, replacing, removing, or
  running tests. Real LLM cases must run and be inspected one at a time.
- `local-llm-architecture`: load before changing the action protocol, prompt,
  context reducer, index-to-model evidence shape, or model responsibility.
- `no-prepost-user-input`: load before changing how task instructions,
  approvals, revision requests, blocker answers, or user permissions enter the
  loop.
- `debug-llm`: load before adding live action-loop evaluation artifacts or
  running the benchmark.

## Mandatory Rules

- A completed progress checkpoint is immutable. Later findings must create a
  new remediation checkpoint with links to the completed checkpoint and new
  evidence; they must never uncheck, reopen, rewrite, or erase the completed
  checkpoint or its original sign-off.
- Keep progress reporting append-only. At every handoff, record the current
  checkpoint, completed evidence, active blocker, next action, and remaining
  dependency-ordered checkpoints before continuing implementation.
- During the architecture-closure checkpoint added on 2026-07-11, run no
  pytest command, benchmark command, live-LLM call, model smoke, or other test
  harness. Use static source, diff, contract, and retained-artifact inspection
  only until the user explicitly authorizes test execution.
- Complete one static failure-mode and provenance review before the next
  production change. Then make one cohesive implementation pass against the
  accepted contract instead of alternating code changes with live cases.
- Before preparing the test harness, record a confidence assessment against
  every failure-mode matrix row. If confidence is below 90%, return to static
  design/code review and repeat the assessment. At 90% or higher, prepare but
  do not execute the deterministic or live-LLM harness. A real-LLM run always
  requires the user's explicit permission.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Use parent-led native subagent execution unless the user explicitly approves
  fallback execution.
- Keep `coding_run` as the sole durable public workflow boundary. The new loop
  is an internal execution engine and must not create a second public run API.
- Use one model controller and one strict JSON action per LLM call. Do not add
  model-native tool calling, a second manager hierarchy, or separate read,
  write, and repair controller prompts.
- Give the model semantic action availability, never raw commands, absolute
  paths, credentials, lock details, execution configuration, or approval
  internals.
- Deterministic code must validate every action before it reads, searches,
  mutates, executes, persists, or renders anything.
- Every edit must target the Phase B managed candidate workspace. Original
  source must remain unchanged throughout generation and verification.
- Repository index results are evidence. They must not become instructions,
  permissions, persona, or final implementation judgment.
- The index must exclude `.git`, environment files, credential-like paths,
  binary content, symlinks, sockets, devices, and any path outside the resolved
  source scope.
- Remove structural exploration caps from the production path. Run-level wall
  time, model-call, context, execution, and no-progress budgets remain required
  safety controls and must end in a typed Phase C blocker rather than silent
  truncation.
- Do not retain a runtime route back to the old `code_reading`, `code_writing`,
  or `code_modifying` model pipelines after cutover.
- Do not use current test topology as a reason to preserve deprecated
  production boundaries. Replace or remove tests when their owned boundary is
  replaced or removed.
- Anti-cheat rule: loop verification must start through the public coding-run
  objective and exercise JSON parsing, deterministic dispatch, persistence,
  candidate state, and terminal projection. Direct handler calls cannot claim
  end-to-end coverage.
- Anti-cheat rule: repository-scale verification must prove discovery of files
  beyond the former 120-file map and former per-programmer file bounds. A
  fixture that directly supplies the target path does not prove index search.
- Anti-cheat rule: delete/rename verification must materialize the candidate and
  final review package from a real baseline. Hand-authored expected diffs alone
  do not prove operation safety.

## Must Do

- Add `coding_agent/code_action_loop/` as the canonical model-facing engine for
  `read_only`, `propose_patch`, and `verify_repair`.
- Add `coding_agent/repository_index/` as the deterministic persistent indexing
  and search boundary.
- Define a closed `coding_action.v1` JSON protocol with `read`, `search`,
  `edit`, `run`, `note`, `finish`, and `block` actions.
- Define objective-specific capabilities without creating different protocols:
  `read_only` allows read/search/note/finish/block; proposal and repair allow
  read/search/edit/note/finish/block; `run` is exposed only when Phase B policy
  says execution is currently available.
- Persist the loop state, validated actions, deterministic observations,
  working notes, candidate revision, index snapshot identity, and budget usage
  under the durable coding-run workspace.
- Build a complete immutable index snapshot for every resolved source identity
  and reuse unchanged indexed content across runs.
- Add a run-local candidate overlay so searches immediately reflect create,
  replace, insert, delete, and rename operations without rebuilding the base
  snapshot.
- Replace repository-map, file-list, manager-wave, programmer-report, and
  fallback-evidence caps in all three coding objectives.
- Add guarded `delete_file` and `rename_file` operations to the canonical patch
  operation vocabulary and review/apply path.
- Convert Phase B execution results and repair failure bundles into normal loop
  observations so the same controller can continue rather than restart a
  separate repair hierarchy.
- Replace `code_verifying/execution_planning.py`'s additive-test LLM with a
  deterministic plan derived only from the validated candidate operation
  records and the controller's already validated semantic `run` request.
  Remove `extract_additive_execution_specs`, `ADDITIVE_EXECUTION_SPEC_PROMPT`,
  `MAX_ADDITIVE_SAFE_TEST_PATHS`, and every PM-route invocation from execution
  planning.
- Use the Phase C benchmark seam as the pre-cutover evidence gate.
- Remove the superseded production model pipelines, prompts, and topology-bound
  models from `code_reading`, `code_writing`, and `code_modifying` after the
  gate passes and the production dispatcher moves to the action loop.
- Align coding-agent README files, architecture references, and lifecycle
  registry with the final boundary.

## Deferred

- Arbitrary shell command execution.
- Package installation or dependency mutation.
- Non-Python build-system expansion beyond Phase B's allowlisted execution
  policy.
- Git commit, branch, push, pull-request, or repository publishing actions.
- Network search inside the loop.
- Private repository authentication changes.
- Binary editing, directory deletion, directory rename, symlink creation, file
  permission changes, and executable-bit changes.
- Multi-repository transactions or cross-source moves.
- Interactive terminal sessions, background processes, and long-running server
  management.
- A user-visible streaming coding console.
- Breadth features reserved for a later Phase E plan.

## Cutover Policy

Overall strategy: bigbang. The isolated benchmark harness is the sole
explicit compatible surface before cutover; legacy ledgers use the one-time
migration stated below.

| Area | Policy | Instruction |
| --- | --- | --- |
| Production coding-run dispatch | bigbang | Route all three new-run objectives to `action_loop_v1` in one change; retain no runtime engine switch or fallback. |
| Evaluation benchmark harness | compatible | Permit `pipeline_v1` and `action_loop_v1` only through the private isolated evaluation entry point and locked v2 artifacts. |
| Legacy nonterminal ledgers | migration | Under the cutover lock, mark them `pipeline_v1_retired`, preserve them read-only, and reject continuation mutations. |
| Repository index data | bigbang | Create only the v1 snapshot store; retain complete pinned snapshots until eligible for reclamation. |
| Tests, prompts, and documentation | bigbang | Replace target-state coverage and remove all topology-owned source, tests, and docs in the same cutover. |

- Phase D remains `draft` until Phase B and Phase C are complete.
- Phase B and Phase C completion evidence is recorded above. The remaining
  draft gate is a frozen, complete 30-case baseline artifact with the scoring
  fields defined in **Benchmark Artifact Contract** below.
- Implementation begins only after this plan is approved and the user
  explicitly authorizes production-code changes.
- The new action loop is first reachable only through an evaluation entry point
  that writes to isolated benchmark run roots.
- Evaluation may compare `pipeline_v1` and `action_loop_v1`, but production
  dispatch continues to use exactly one implementation during evaluation.
- The production cutover requires the benchmark gate in this plan, completed
  deterministic target-state verification, reviewed migration inventory, and
  an independent code review with no unresolved critical or high findings.
- At cutover, all three objective routes move to `action_loop_v1` in one change.
- The same cutover removes imports, prompts, route branches, and tests owned
  only by the superseded model pipelines. Shared deterministic behavior is
  moved to its final owner before old packages are removed.
- Runtime fallback, dual-write state, alias actions, old-to-new model adapters,
  and environment switches that restore `pipeline_v1` are forbidden.
- If the benchmark gate fails, production remains on `pipeline_v1`, Phase D
  stays `in_progress`, and remediation occurs in the evaluation-only engine.
  The failed engine is not partially deployed.
- The evaluation-only entry point is
  `coding_run/evaluation.py::run_evaluation_coding_run`. It accepts the same
  durable request shape as `start_coding_run`, requires an explicit
  `engine_id`, and writes only below the supplied benchmark run root. It is
  unavailable from the public coding-agent exports, background worker, L2d,
  and adapter paths.

### Executive Architecture Decision (2026-07-11)

- Stage 4 is entirely evaluation-only. It extends the existing private
  `coding_run/evaluation.py` boundary with start and continuation entrypoints
  that use the final internal action-loop start/resume functions and preserve
  approval and blocker-resume semantics.
- The public `start_coding_run` and `continue_coding_run` APIs remain on
  `pipeline_v1` until the Stage 5 two-engine comparison passes. There is no
  second non-production runtime boundary and no public engine selector.
- Public big-bang cutover, legacy-ledger retirement, pipeline deletion,
  post-cutover public integration proof, static scans, documentation updates,
  and independent code review occur only after the comparison gate passes.
- The Stage 1–3 checkmarks are reopened because the initial helper-level tests
  did not prove the benchmark, persistent-index, candidate durability, or
  canonical operation/apply contracts required by this plan.

## Target State

```text
accepted task / direct coding request
  -> coding_run durable objective
  -> source resolution + Phase B candidate preparation
  -> persistent repository-index snapshot
  -> action_loop controller
       -> strict JSON action
       -> deterministic action validation
       -> read/search/edit/run executor
       -> durable observation + context reduction
       -> next controller turn
  -> finish or typed block
  -> deterministic patch/review assembly
  -> Phase B approval + verification
  -> same action loop for repair observations when needed
  -> coding_run result projection + Phase C continuation context
```

The loop is generic at the protocol level and capability-scoped at runtime.
Read-only work uses the same search, observation, note, and finish machinery
without receiving edit or run permissions. Proposal and repair work operate on
the managed candidate. Execution is an optional deterministic capability, not
a command channel.

### Post-Cutover Public Boundary

- `start_coding_run`, `continue_coding_run`, and `get_coding_run` remain the
  only public direct coding APIs and the only background-worker entrypoint.
  `coding_agent/__init__.py` exports only those run APIs for coding workflow
  execution; direct `answer_code_question`, `propose_code_change`, and
  `verify_and_repair_code_change` imports are removed from public exports and
  production callers.
- `coding_run/supervisor.py` dispatches every new `read_only`, `propose_patch`,
  and `verify_repair` run to `code_action_loop/supervisor.py`. The direct
  modules remain only as Phase B deterministic source, candidate, patch, apply,
  and execution primitives invoked by the loop; they have no route that calls
  a manager/programmer LLM or restarts a proposal tree.
- `coding_agent/supervisor.py`, background-worker callers, and all tests move
  to the run API. The post-cutover static scan requires zero production imports
  of `code_reading`, `code_writing`, or `code_modifying`, except deterministic
  helpers explicitly moved to their final owner before package deletion.

## Design Decisions

### 1. One Controller Owns Semantic Next-Action Choice

- One LLM controller decides the next semantic action from task state and
  observations.
- The controller does not delegate to product-manager or programmer LLMs.
- Every LLM call returns exactly one JSON object.
- The controller may explore any number of repository files permitted by the
  run-level resource budget; there is no per-wave, per-agent, per-report, or
  per-file exploration allowance.
- The controller is responsible for deciding when evidence is sufficient,
  which candidate edit to attempt, whether available verification is useful,
  and whether to finish or issue a typed blocker.
- Deterministic code is responsible for deciding whether the requested action
  is valid and executable.

### 2. Strict JSON, Not Native Tool Calling

- Parse raw model output through the existing strict JSON extraction boundary.
- Validate the parsed object against the closed action schema and the current
  objective's allowed actions.
- A malformed JSON response consumes one action turn and produces a bounded
  `invalid_json` observation for the next controller call.
- A parsed but invalid action consumes one action turn and produces a bounded
  `invalid_action` observation naming schema errors and currently allowed
  actions.
- Do not invoke a second LLM to repair JSON.
- Three consecutive invalid outputs terminate in a typed
  `controller_contract_failure` blocker with resume target `retry_loop`.
- `reason` and `working_note` request concise operational summaries only. The
  prompt must not request hidden chain-of-thought or unrestricted reasoning
  transcripts.

### 3. Durable Loop State

- Persist loop artifacts beneath:
  `<coding_run_root>/action_loop/`.
- Use:
  - `state.json` for protocol version, objective, status, budgets, candidate
    revision, base index snapshot, overlay revision, and terminal result;
  - `actions.jsonl` for raw output hash, parsed action, validation result,
    timing, and action outcome;
  - `observations.jsonl` for prompt-safe deterministic observations;
  - `working_notes.json` for bounded model-authored task state;
  - `context_manifest.json` for the exact items included in each model call.
- Treat all files as resumable run state under Phase C's per-run lock.
- Append an action event before dispatch and an observation event after the
  deterministic outcome. Recovery must detect an action without a terminal
  observation and reconcile it without replaying a mutation blindly.
- Bind every mutating action to the current candidate revision. A successful
  mutation advances the revision exactly once.

### 4. Persistent Repository Index

- Store indexes beneath:
  `<coding_workspace_root>/repository_indexes/<source_identity_hash>/`.
- Use Python's standard `sqlite3` with FTS5 as the canonical local store.
- Separate immutable snapshots by:
  `sha256(source_identity + source_manifest_digest + exclusion_policy_version + index_schema_version)`.
- The source manifest contains sorted repo-relative paths, content SHA-256,
  byte size, and safe-text classification for all in-scope regular files.
- A snapshot is queryable only after an atomic `complete` marker is committed.
  Incomplete builds remain resumable and must not be presented as complete.
- Reuse chunks and symbols for unchanged content hashes. Build new or changed
  rows incrementally, then atomically publish the new snapshot.
- `storage.py` retains every complete snapshot referenced by any non-archived
  coding-run ledger. Its reclamation pass may remove only an unpinned complete
  snapshot after a full source-identity scan under the source lock; it never
  deletes a `building` snapshot, a pinned snapshot, or a snapshot selected by
  an active cursor.
- Index all safe text files in source scope. There is no maximum indexed-file
  count and no early stop after a fixed number of files.
- Stream large text files into bounded line-aware chunks so index construction
  does not require loading an entire repository or file into memory.
- Initial symbol extraction supports Python through `ast`. All safe text still
  receives full-text and path indexing when no language-specific extractor is
  available.
- Keep the index deterministic and local. The model never receives the
  database path, raw SQLite rows, or unrestricted dump of repository content.

### 4a. Versioned Index Exclusion And Resource Policy

- `repository_index/identity.py` defines `RepositoryIndexExclusionPolicyV1`
  before any file is hashed, chunked, or indexed. It excludes `.git`, virtual
  environments, cache directories, generated repository-index and coding-run
  artifacts, `test_artifacts`, symlinks, devices, sockets, binaries, files
  outside the resolved source scope, `.env` and `.env.*`, and basename/path
  segments matched case-insensitively by `credential`, `secret`, `token`,
  `password`, `private_key`, or `id_rsa`.
- The same policy streams the complete otherwise-safe text file before any
  chunk is persisted and scans every byte for common credential assignments
  and PEM private-key markers. A match anywhere, including the middle of a
  large file, excludes the complete file and records only its repo-relative
  path plus exclusion reason in the manifest. The index, overlay,
  observations, and `working_notes.json` must never retain excluded content.
- `context_budget.py` owns the explicit non-exploration limits:
  `MAX_INDEX_FILE_BYTES`, `MAX_INDEX_CHUNK_BYTES`, `MAX_SEARCH_RESULTS_PER_PAGE`,
  `MAX_SEARCH_EXCERPT_CHARS`, `MAX_READ_LINES`,
  `MAX_REGEX_QUERY_CHARS`, `REGEX_SEARCH_TIMEOUT_MS`, and
  `MAX_INDEX_STORAGE_BYTES`. An exceeded byte, disk, or build-time budget
  yields a typed `index_resource_exhausted` blocker with the exact resource;
  it never silently drops an otherwise eligible file. These are resource and
  prompt-output limits, not file-count, wave, report, or exploration limits.
- A cursor encodes `snapshot_id`, `overlay_revision`, search mode, normalized
  query, normalized path glob, and last deterministic sort tuple. Any mismatch
  returns `stale_cursor` with no result rows.

### 5. Index Schema And Search Contract

The SQLite database owns these canonical entities:

```text
snapshot(snapshot_id, source_identity_hash, manifest_digest,
         schema_version, exclusion_policy_version, status, created_at)
file(snapshot_id, file_id, repo_path, content_sha256, byte_size, language,
     line_count)
chunk(snapshot_id, chunk_id, file_id, repo_path, start_line, end_line,
      content)
chunk_fts(chunk_id UNINDEXED, snapshot_id UNINDEXED, repo_path UNINDEXED,
          content)
symbol(snapshot_id, symbol_id, file_id, qualified_name, symbol_kind,
       start_line, end_line, signature)
import_edge(snapshot_id, edge_id, file_id, imported_name, target_repo_path)
```

- `snapshot_id` is part of every table's primary or unique key. A file path is
  unique only within one snapshot. The builder writes rows only for a
  `building` snapshot and changes that snapshot to `complete` in the same
  SQLite transaction that records its completion marker.
- `chunk_fts` is an FTS5 projection maintained by the repository index store.
  It duplicates only the chunk identifier, snapshot identifier,
  repo-relative path, and safe-text content needed for FTS; it is not an
  external-content FTS table.

- `search` supports `literal`, `regex`, `symbol`, and `path` modes through one
  deterministic interface.
- FTS handles ranked literal search. Regex executes in one standard-library
  worker process over the deterministic chunk order. The parent terminates the
  worker at `REGEX_SEARCH_TIMEOUT_MS = 500`, returns no partial rows, and
  records a `regex_timeout` observation. Symbol and path modes use their
  dedicated tables/indexes.
- Search results return only:
  `repo_path`, `start_line`, `end_line`, `match_kind`, `symbol`, `excerpt`,
  `content_sha256`, and `candidate_revision`.
- Order results deterministically by match quality, exactness, path, and line.
- Return bounded excerpts per action, while allowing the controller to issue
  further search/read actions. Result paging uses an opaque deterministic
  cursor bound to the same snapshot and overlay revision.
- Search result count per action is a context-output bound, not an exploration
  bound. The controller can page or refine until the run budget ends.
- `read` with `start_line` defaults `end_line` to
  `min(start_line + 199, file_line_count)` and rejects spans above 500 lines.
  The bounded read is a prompt-output bound; subsequent reads may retrieve
  further ranges until the run budget ends.

### 6. Candidate Overlay

- Keep the immutable source snapshot unchanged during a coding run.
- `code_action_loop/state.py` owns one long-lived candidate root at
  `<coding_run_root>/candidate/source/`. Source-backed runs create it once from
  the resolved Phase B source before the first action. Source-free runs create
  the same empty root with `source_identity_hash = sha256("source_free.v1" +
  run_id)` and the empty-manifest digest; its base snapshot contains zero
  files. Both forms use the same candidate, overlay, review, and preflight
  contracts.
- Maintain a run-local overlay index for every created, modified, deleted, or
  renamed path in that candidate. `state.json` records the authoritative
  operation journal: `operation_id`, expected and resulting candidate
  revisions, path/hash preconditions, operation kind, and state
  `prepared|candidate_written|overlay_written|committed|rolled_back`.
- `actions.py` performs an edit in this fixed order while holding the Phase C
  sorted source/run locks: append `prepared`; materialize the candidate change
  with a same-directory temporary file; fsync and atomically replace/remove;
  append `candidate_written`; update the SQLite overlay in one transaction;
  append `overlay_written`; advance the candidate revision and append
  `committed`. The action event is idempotent by `operation_id`.
- Recovery reads the journal before accepting another action. It recomputes the
  candidate path hashes and overlay rows: a matching committed state is reused;
  a candidate-written state completes only the missing overlay write; an
  overlay-written state completes only the missing ledger commit; any mismatch
  rolls the candidate back from the per-operation backup before recording a
  typed `candidate_recovery_failed` blocker. It never replays a model mutation
  from raw action text.
- A deleted source path becomes a tombstone.
- A rename creates a source tombstone and a target live entry linked by one
  operation id.
- Search merges the base snapshot with the overlay by excluding every
  overlay-owned base path, then adding current overlay rows.
- Read always resolves against the current candidate view for proposal and
  repair objectives, and against the immutable source snapshot for read-only
  objectives.
- Search and read observations include the snapshot id and candidate revision
  so stale evidence is detectable.

### 7. Delete And Rename Are Canonical Edit Operations

- Add `delete_file` and `rename_file` beside existing canonical patch
  operations.
- `delete_file` requires:
  - a normalized repo-relative source path;
  - an existing in-scope regular safe-text file;
  - an exact expected source SHA-256;
  - the current candidate revision;
  - absence from the protected verification-path set.
- `rename_file` requires:
  - normalized source and target repo-relative paths;
  - an existing in-scope regular safe-text source;
  - an absent target;
  - exact expected source SHA-256;
  - the current candidate revision;
  - both paths within the same source scope;
  - absence of both paths from the protected verification-path set.
- Rename is content-preserving. A content change after rename is a separate
  edit action against the target path.
- Directory, symlink, binary, device, socket, cross-source, and case-collision
  delete/rename operations are rejected deterministically.
- On case-insensitive filesystems, reject any rename whose normalized source
  and target collide. A future explicit case-only rename contract is outside
  this phase.
- Apply candidate mutations through same-directory temporary materialization
  and atomic replacement/removal steps while holding the Phase C source/run
  locks.
- Represent delete as `/dev/null` target diff semantics and rename as explicit
  old/new path metadata plus content identity in the review package.
- Bind delete/rename operations into the proposal digest, candidate revision,
  approval evidence, execution plan, and final applied result.
- `code_patching/models.py` defines canonical operation records with
  `operation_id`, `kind`, `source_path`, `target_path`,
  `expected_source_sha256`, `expected_candidate_revision`,
  `result_sha256`, and `content_sha256`. Create records use
  `expected_source_sha256 = null` and require an absent target; delete records
  set `target_path = null` and render `--- a/<path>` plus `+++ /dev/null`;
  rename records carry both paths, preserve identical
  `expected_source_sha256` and `content_sha256`, and render an explicit
  review-only `rename from` / `rename to` header with no content hunk.
- The canonical proposal digest is SHA-256 over the UTF-8 canonical JSON array
  of operation records in operation order, including both paths and all hash
  and revision preconditions. `PatchArtifact`, `ChangedFileSummary`, review
  rendering, approval binding, execution plan, apply result, and rollback
  evidence all carry that digest and every affected path. Apply validates every
  record against the candidate revision before any target operation, stages a
  complete managed-copy result, and atomically publishes only that managed
  result. It records the same ordered operation records as final apply
  evidence.

### 8. Run Is A Semantic Action

- The model may request `run` only when the current objective and Phase B
  policy expose it.
- `run.args` contains a semantic profile, optional repo-relative targets, and a
  short intent. It never contains a command string, shell syntax, environment
  overrides, or working-directory selection.
- Phase B's deterministic execution planner maps the semantic request to
  allowlisted derived-base or focused verification.
- The runner executes only in the managed candidate and returns the structured
  Phase B execution result/failure bundle as an observation.
- A currently unavailable run request returns `action_unavailable` without
  starting a process.
- A run action cannot authorize its own execution, expand tool policy, or
  bypass proposal-bound approval/revision checks.
- The model receives `run` in its semantic capability list only when the
  deterministic loop state records a current Phase B proposal digest,
  candidate revision, trusted execution-plan binding, and enabled preflight or
  approved-verification state. The model receives only `derived_base` or
  `focused` and a short semantic description; it never receives flag values,
  argv, selectors, lock data, or approval provenance.

### 9. Finish And Repair Use The Same Loop

- `finish` for `read_only` must include a source-grounded answer and evidence
  references already observed in the loop.
- `finish` for proposal/repair must include a concise change summary,
  acceptance-criterion disposition, and remaining known limitations.
- Deterministic finalization revalidates candidate operations, builds the
  review package, checks patch/candidate alignment, and invokes available
  Phase B preflight verification.
- A finalization or verification failure becomes a normal structured
  observation and returns control to the same loop while repair budget remains.
- Approved verification failure reopens the durable loop at the same candidate
  lineage with the Phase B failure bundle and a new candidate revision.
- Repair must not invoke `code_modifying` or recreate the previous PM/programmer
  tree.

### 10. Removal Of Exploration Caps

- Remove these production exploration constraints from all three objectives:
  - `MAX_REPOSITORY_MAP_FILES`;
  - `MAX_PROGRAMMERS_PER_WAVE`;
  - `MAX_PROGRAMMER_WAVES`;
  - `MAX_PROGRAMMER_REPORTS_PER_PM`;
  - `MAX_FILES_PER_PROGRAMMER`;
  - `MAX_WRITE_FALLBACK_EVIDENCE_FILES`;
  - equivalent fixed slices of repository evidence introduced elsewhere.
- Remove the runtime structures that require those constants rather than
  setting them to larger numbers.
- Do not materialize a full repository map into the prompt. The index is the
  discovery surface; search/read observations enter context on demand.
- Keep explicit run-level resource controls described below. These controls
  prevent runaway work but do not predetermine which or how many files the
  model may inspect.

## Contracts And Data Shapes

### `coding_action.v1`

```json
{
  "schema_version": "coding_action.v1",
  "action_id": "model-generated-id",
  "action": "read|search|edit|run|note|finish|block",
  "reason": "short operational reason",
  "args": {},
  "working_note": "optional bounded state update"
}
```

- `schema_version`, `action_id`, `action`, `reason`, and `args` are required.
- Reject unknown top-level keys and unknown action-specific keys.
- `action_id` is diagnostic only. Deterministic persistence assigns the
  authoritative sequence number and operation id.
- Bound `reason` to 600 characters and `working_note` to 2,000 characters.

### Action Arguments

```text
read:
  repo_path
  start_line | symbol
  end_line?                 # required with start_line unless deterministic default applies

search:
  mode                      # literal | regex | symbol | path
  query
  path_glob?
  cursor?

edit:
  operation                 # create_file | replace_anchor | insert_before |
                            # insert_after | replace_file_small |
                            # delete_file | rename_file
  repo_path                 # source path for delete/rename
  target_path?              # rename only
  expected_sha256?          # required for existing-file operations
  expected_candidate_revision
  anchor?                   # anchor operations only
  replacement?              # create/replace/insert operations only

run:
  profile                   # derived_base | focused
  targets?                  # repo-relative semantic targets
  intent

note:
  completed
  remaining
  assumptions

finish:
  summary
  acceptance_criteria
  evidence_refs
  known_limitations

block:
  blocker_type
  question
  options
  resume_target
  blocking_evidence_refs
```

- Existing edit operation names may be normalized once during this big-bang
  migration. One canonical vocabulary must remain after cutover.
- Do not support aliases for old and new operation names concurrently.
- `create_file` requires an absent normalized `repo_path`,
  `expected_candidate_revision`, and nonempty `replacement`; it rejects a
  path already visible through the base snapshot or overlay. Every other edit
  requires `expected_sha256` for the current candidate view and the exact
  `expected_candidate_revision`. Anchor operations require one exact anchor
  match; zero or multiple matches return a stale observation. `replacement`,
  `anchor`, `query`, and `working_note` are bounded by the context-output
  constants in `context_budget.py`; no action may carry an unbounded source
  file or diff.

### `coding_observation.v1`

```json
{
  "schema_version": "coding_observation.v1",
  "sequence": 12,
  "action_sequence": 11,
  "outcome": "ok|rejected|failed|unavailable|stale",
  "kind": "read_result",
  "summary": "prompt-safe deterministic summary",
  "evidence": [],
  "candidate_revision": 4,
  "index_snapshot_id": "sha256:...",
  "overlay_revision": 4,
  "budget_remaining": {},
  "created_at": "RFC3339 timestamp"
}
```

- Observation evidence is typed by kind and schema-validated before prompt
  inclusion.
- Store full safe deterministic output in the run artifact when prompt
  truncation is necessary; include its artifact reference in the observation.
- Never store or expose secrets, raw environment values, absolute host paths,
  or unrestricted process environment.

### Blocker And Resume Contract

- The controller's `block.args.blocker_type` is a closed enum:
  `needs_user_input|environment|scope|safety|budget|controller_contract_failure|index_resource_exhausted|candidate_recovery_failed`.
  `options` is a bounded list of prompt-safe strings. The controller may
  describe the unresolved need, but deterministic code chooses the only valid
  `resume_target` from `replan_proposal|retry_verification|retry_loop|none`
  using the current objective and loop state.
- `coding_run/models.py` extends the Phase C blocker schema and its validators
  with those exact two operational failure kinds and `retry_loop`; all other
  controller-provided values are rejected as an `invalid_action` observation.
  A `budget`, `controller_contract_failure`, `index_resource_exhausted`, or
  `candidate_recovery_failed` blocker always has `resume_target = retry_loop`.
- A `respond_to_blocker` continuation remains a public typed action. It stores
  the user's answer verbatim as the next loop's dynamic input alongside the
  durable blocker and original goal. The controller semantically decides the
  next action; deterministic code performs no keyword classification,
  rewriting, or acceptance inference over that answer.

### Model-Facing Payload Contract

- `code_action_loop/prompts.py` has one static system prompt containing the
  controller role, closed action schema, semantic capability descriptions,
  bounded-output rules, and positive action-selection procedure. It contains
  no migration labels, module names, raw path roots, locks, database schema,
  command text, deployment flags, or approval internals.
- Each controller call receives one dynamic payload with the goal, plain
  acceptance criteria, capability names, source identity digest, candidate
  revision, bounded working notes, changed-path summary, unresolved failures,
  and schema-validated prompt-safe observations. It never receives raw source
  dumps, excluded file contents, raw process output, full SQLite rows, or
  numeric operational telemetry without deterministic semantic labels.
- `reason` and `working_note` are operational summaries, not hidden reasoning.
  `context.py` deterministically renders the payload and records its manifest;
  `prompts.py` has a runtime `.format(...)` render test before live calls.
- The controller is the only post-source-resolution coding LLM and
  `code_action_loop/supervisor.py` owns the new `CODING_AGENT_ACTION_LOOP_LLM`
  route configuration from `config.py`. The upstream
  `code_fetching/source_intake.py` remains the sole allowed
  `CODING_AGENT_PM_LLM` caller for source-resolution semantics. Remove the
  `CODING_AGENT_PROGRAMMER_LLM` configuration and every use of it at cutover.

### Index Identity

```text
RepositoryIndexIdentityV1:
  source_identity_hash
  source_manifest_digest
  exclusion_policy_version
  index_schema_version
  snapshot_id
  status                       # building | complete | invalid
```

- The coding run pins one complete base snapshot id.
- A changed resolved source identity or source manifest creates a new snapshot.
- A run must never silently switch base snapshots after the first model action.

## LLM Call And Context Budget

- Default maximum controller turns per run segment: 48.
- Default maximum wall time per run segment: 30 minutes, excluding time spent
  waiting for explicit user approval or blocker response.
- Default maximum `run` actions per run segment: 8.
- Maximum consecutive invalid controller outputs: 3.
- Maximum consecutive no-progress actions with the same normalized
  action/observation signature: 3.
- Maximum rendered model context per call: 50,000 characters.
- Maximum single read/search observation included in the next prompt: 8,000
  characters; retain the full safe artifact outside the prompt.
- Maximum working-notes payload: 12,000 characters after deterministic schema
  rendering.
- Include the current task, acceptance criteria, objective capabilities,
  source/index identity, candidate revision, working notes, unresolved
  failures, changed-path summary, and newest observations.
- Evict oldest resolved observations first. Never evict current task,
  acceptance criteria, action schema, permissions, unresolved blocker/failure,
  current candidate revision, or current changed-path summary.
- Context reduction is deterministic. It may truncate or omit old evidence but
  must not reinterpret model notes or user instructions.
- Reaching a wall-time, turn, run-action, or no-progress budget emits a typed
  Phase C blocker with the exact exhausted budget, latest evidence, and resume
  target. It must not fabricate completion or silently drop work.
- Trusted configuration may change resource-budget values. The model cannot
  change them, and no configuration may reintroduce per-file, per-wave,
  per-agent, repository-map, or report-count exploration caps.

## Data Migration

- Add `action_loop_version`, pinned index identity, candidate/overlay revision,
  and loop artifact references to the durable coding-run model in one schema
  update.
- Existing unfinished `pipeline_v1` runs do not migrate into an invented loop
  history and do not receive a new continuation path. Before production
  cutover:
  - allow already-running work to reach a stable terminal or blocked state;
  - reject new continuation mutations during the cutover lock window;
  - preserve old artifacts as read-only audit evidence;
  - mark each nonterminal legacy ledger `blocked` with the typed
    `pipeline_v1_retired` blocker and `resume_target = none`.
- Legacy ledgers remain readable through `get_coding_run` and reject every
  mutating `continue_coding_run` request. A user starts a fresh
  `action_loop_v1` run with an explicit request; no code copies a legacy goal,
  approval, candidate, action, or execution-plan state into the new ledger.
- The new ledger carries `engine_id = action_loop_v1` and
  `ledger_schema_version = coding_run.v4`. `ledger.py` rejects a missing or
  unsupported ledger version before dispatch. This is a one-way cutover, not a
  compatibility adapter or state migration.
- Build repository indexes lazily on first post-cutover use and reuse them
  afterward. Do not eagerly scan unrelated repositories during deployment.
- Stamp all index, loop, action, observation, and patch-operation artifacts with
  explicit schema versions.

## Benchmark Artifact Contract

- Before implementation dispatch changes, replace the status-only Phase C
  benchmark artifact with `coding_agent_benchmark.v2`. Freeze exactly 30 rows
  in `tests/fixtures/coding_agent_benchmark/cases.jsonl`; each row contains
  `case_id`, `objective_type`, literal task text, fixture reference, fixture
  source-manifest digest, route-config digest, expected terminal state,
  acceptance checks, hard safety gates, and five integer rubric scores:
  task progress, repository grounding, repair quality, safety/privacy, and
  authorization. The manifest itself is SHA-256 locked in
  `test_artifacts/coding_agent_benchmark/manifest-lock.json` before the first
  baseline case.
- Each rubric score is `0`, `1`, or `2`: `0` fails the named dimension, `1`
  reaches a useful but incomplete result, and `2` satisfies the fixed case
  acceptance target. The stored judgment note cites trace evidence for every
  non-`2` score. Safety/privacy and authorization scores are `0` whenever a
  related hard gate fails; their category averages are compared without
  rounding.
- `scripts/run_coding_agent_benchmark.py` runs exactly one named case through
  `run_evaluation_coding_run` with `engine_id` set to `pipeline_v1` or
  `action_loop_v1`. It writes one immutable result below
  `test_artifacts/coding_agent_benchmark/<engine_id>/<case_id>/result.json`.
  Each result records the locked manifest digest, source digest, objective,
  redacted route/config digest, elapsed time, model-call count, token use when
  supplied, terminal state, acceptance and hard-gate outcomes, five rubric
  scores, raw output hash, parsed action/observation trace paths, context
  manifest paths, and a required human/agent judgment note.
- The parent runs all 30 `pipeline_v1` cases one at a time, inspects each trace,
  and freezes their result paths before `action_loop_v1` runs. Both engine sets
  use the identical locked manifest, fixture source digest, task text,
  objective, and route-config digest. A changed digest invalidates the compared
  pair and requires a fresh baseline case.
- The fixed per-case benchmark timeout is 600 seconds. A changed timeout
  invalidates every baseline result that reached the previous timeout and
  requires that case to be rerun before Stage 1 can be signed off.
- The offline comparator rejects missing, duplicate, unlocked, non-passing
  hard-gate, or mismatched pairs. It calculates end-state success as passing
  terminal state plus all acceptance checks; it calculates each rubric average
  by category. It emits a single comparison artifact at
  `test_artifacts/coding_agent_benchmark/comparisons/action_loop_v1_vs_pipeline_v1.json`.
  Cutover requires an overall action-loop success rate at least 10 percentage
  points higher, no category decline in safety/privacy or authorization average,
  and zero failures for source mutation, stale index, command policy, approval,
  repository-scale, delete, or rename hard gates. Median model calls and wall
  time are reported only.

## Cutover Policy Enforcement

| Surface | Evaluation policy | Production cutover enforcement | Rollback rule |
| --- | --- | --- | --- |
| Public coding-run API | Private evaluation entry point only | New public runs use `action_loop_v1` through the existing three run APIs | Before cutover, leave production dispatch on `pipeline_v1`; after cutover, remediate forward with no runtime engine switch. |
| Legacy ledgers | Read-only audit artifacts | Block nonterminal `pipeline_v1` ledgers with `pipeline_v1_retired` | No state migration or continuation adapter. |
| Benchmark | Both engine IDs use the locked v2 manifest | Comparator artifact and all hard gates pass | A failed evaluation blocks the cutover. |
| Repository indexes | Evaluation roots only | Create lazily beneath the approved coding workspace | Delete only incomplete index directories; complete snapshots remain readable until no ledger pins them. |
| Model pipelines | `pipeline_v1` remains only for isolated comparison | Remove manager/programmer routes, imports, prompts, packages, and topology-only tests in the same change | Production rollback is unavailable after deletion; restore only by a separately approved plan. |
| Docs and tests | Target-state tests accompany implementation | Update listed ICD/READMEs and remove superseded topology-owned tests | A documentation or scan failure blocks completion and requires forward remediation. |

- The responsible execution agent follows the selected policy for every area
  in the policy matrix.
- A `bigbang` area deletes or rewrites legacy references at cutover; it does
  not preserve an adapter, fallback, dual path, or old public state shape.
- The evaluation harness preserves both engines only within its explicit
  `compatible` surface and must never expose its selector to runtime callers.
- The legacy-ledger migration follows the exact locked terminalization steps in
  **Data Migration**. A change to any policy requires user approval before
  implementation.

## Change Surface

### Add

- `src/kazusa_ai_chatbot/coding_agent/code_action_loop/`
  - `models.py`
  - `prompts.py`
  - `parser.py`
  - `context.py`
  - `actions.py`
  - `state.py`
  - `supervisor.py`
  - `README.md`
- `src/kazusa_ai_chatbot/coding_agent/repository_index/`
  - `models.py`
  - `identity.py`
  - `builder.py`
  - `storage.py`
  - `search.py`
  - `overlay.py`
  - `regex_worker.py`
  - `README.md`
- `src/kazusa_ai_chatbot/coding_agent/safety.py`
- `tests/test_coding_agent_phase_d_repository_index.py`
- `tests/test_coding_agent_phase_d_action_loop_contracts.py`
- `tests/test_coding_agent_phase_d_candidate_recovery.py`
- `tests/test_coding_agent_phase_d_patch_operations.py`
- `tests/test_coding_agent_phase_d_coding_run_integration.py`
- `tests/test_coding_agent_phase_d_benchmark_contracts.py`

### Modify

- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`
  - route all three objectives through the action-loop engine;
  - remove capped proposal fallback evidence assembly.
- `src/kazusa_ai_chatbot/coding_agent/models.py`
  - remove the manager/programmer background-operation vocabulary and retain
    only source and run types still owned after cutover.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/patcher.py`
  - replace writing-model input imports and programmer-route trace metadata
    with canonical operation-record materialization.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_validation.py`
  - consume shared safety predicates from `coding_agent/safety.py`.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/execution_planning.py`
  - retain only deterministic proposal-bound execution-plan derivation; remove
    the additive-test LLM path and fixed test-path slice.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`
- `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py`
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`
  - persist and resume action-loop state through the existing run API.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/evaluation.py`
  - own the private benchmark-only engine selector and isolated artifact root.
- `src/kazusa_ai_chatbot/coding_agent/__init__.py`
  - expose only the canonical public coding-run workflow APIs after cutover.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/models.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_operations.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_validation.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/patcher.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py`
  - add delete/rename validation, candidate mutation, review rendering, and
    apply semantics.
- `src/kazusa_ai_chatbot/coding_agent/code_executing/`
  - accept only semantic execution plans produced by Phase B and return
    structured loop observations.
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/`
  - return failures to the same durable loop and preserve proposal/revision
    binding.
- `src/kazusa_ai_chatbot/coding_agent/context_budget.py`
  - host shared deterministic context bounds without file-count exploration
    caps.
- `scripts/run_coding_agent_benchmark.py`
- `tests/fixtures/coding_agent_benchmark/cases.jsonl`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`
- `README.md`
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/coding_agent/README.md`
- `src/kazusa_ai_chatbot/coding_agent/coding_run/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_executing/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_verifying/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md`
- `docs/SUBAGENT_INTERFACES.md`

### Remove At Cutover

- Model orchestration, prompts, and topology-only models under:
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/`;
  - `src/kazusa_ai_chatbot/coding_agent/code_writing/`;
  - `src/kazusa_ai_chatbot/coding_agent/code_modifying/`.
- Move genuinely shared deterministic validation to its final owner before
  removing these packages.
- `coding_agent/safety.py` is the final owner of safe repo-relative-path,
  binary-path, and secret-path predicates. `repository_index/identity.py` and
  `code_patching/patch_validation.py` import only this module.
- Delete `src/kazusa_ai_chatbot/coding_agent/file_agent.py` and
  `src/kazusa_ai_chatbot/coding_agent/external_evidence.py`; their
  manager-programmer and external-evidence roles are deferred and have no
  action-loop owner.
- Remove `repository_map.py` and all fixed exploration-cap constants/usages.
- Remove tests and documentation whose only owned behavior is the superseded
  manager/programmer topology.

## Overdesign Guardrail

- Build one controller, one action schema, one observation schema, one index
  implementation, and one candidate overlay.
- Do not add a planner LLM, tool-selection LLM, critic LLM, summarizer LLM, or
  index-routing LLM.
- Do not create separate action protocols for read-only, proposal, and repair.
- Do not add a vector database, embedding service, daemon, message broker, or
  external search service. SQLite FTS5 plus deterministic symbol/path search
  and the one-shot standard-library regex worker is the Phase D index.
- Do not generalize delete/rename into arbitrary filesystem operations.
- Do not generalize semantic `run` into shell access.
- Do not preserve the old pipeline as a fallback layer.
- A proposed abstraction must be required by at least two target-state owners
  in this plan or remain local to its owner.

## Agent Autonomy Boundaries

- Allowed implementation autonomy:
  - choose internal helper names and private dataclass decomposition;
  - choose deterministic SQLite indexes and query plans that preserve the
    stated schema/behavior;
  - move shared deterministic helpers directly to their final owner;
  - replace topology-shaped tests with target-state tests;
  - fix scoped defects discovered in new code when they do not alter public
    contracts or permissions.
- Must stop and ask the user before:
  - expanding execution beyond Phase B's allowlisted policy;
  - changing the public coding-run objective vocabulary;
  - adding network access, dependency installation, git publishing, or private
    authentication;
  - indexing paths excluded by this plan;
  - weakening approval, proposal digest, revision, source-scope, or lock
    validation;
  - changing benchmark thresholds or accepting a failed cutover gate;
  - preserving a runtime compatibility path to `pipeline_v1`.

## Implementation Order

1. Record the completed Phase B/C plan paths above. Add the v2 benchmark tests
   in `tests/test_coding_agent_phase_d_benchmark_contracts.py`, run them, and
   record their expected pre-implementation failure because the v2 manifest,
   comparator, and evaluation entry point do not exist.
2. Freeze the v2 30-case manifest and run every `pipeline_v1` case separately.
   Record the inspected baseline result paths and manifest lock; do not change
   production dispatch in this step.
3. Add the failing repository-index tests
   `test_complete_snapshot_is_pinned_and_queryable_only_after_publish`,
   `test_index_exclusion_policy_never_indexes_secret_like_content_in_a_large_file_middle`,
   `test_regex_timeout_returns_no_partial_rows`, and
   `test_stale_cursor_is_rejected`. Implement only
   `repository_index/{models,identity,builder,storage,search,regex_worker}.py`
   until those tests pass.
4. Add the failing candidate-recovery tests
   `test_overlay_hides_tombstone_and_exposes_rename_target` and
   `test_recovery_completes_or_rolls_back_one_journaled_mutation`. Implement
   `repository_index/overlay.py` and `code_action_loop/state.py`; rerun only
   those focused tests before touching patch or run integration.
5. Add the failing patch-operation tests
   `test_delete_and_rename_records_bind_digest_review_and_apply` and
   `test_create_and_existing_edits_enforce_hash_and_revision_preconditions`.
   Update only the named `code_patching` files until the managed-candidate
   tests pass.
6. Add the failing action-loop tests
   `test_parser_rejects_unknown_keys_and_invalid_capabilities`,
   `test_invalid_output_three_strike_blocks`,
   `test_blocker_resume_preserves_verbatim_user_answer`, and
   `test_prompt_payload_excludes_operational_and_secret_data`. Add
   `test_run_planning_uses_no_secondary_llm_or_fixed_test_slice`. Implement the
   action-loop models, parser, context, prompts, actions, and supervisor.
7. Add the failing public integration tests
   `test_all_objectives_enter_action_loop_through_coding_run` and
   `test_legacy_pipeline_ledger_is_read_only_after_cutover`. Wire the existing
   run API, ledger v4, evaluation entry point, Phase B execution primitive, and
   Phase C blockers only after all focused module tests pass.
8. Run the deterministic suites named below. Add or update only target-state
   integration tests, then inventory and remove topology-owned tests and
   packages in the same cutover change.
9. Run the real-LLM controller cases individually, inspect and record every
   artifact, then run every locked `action_loop_v1` benchmark case individually
   and produce the comparator artifact.
10. After the benchmark gate and independent code review pass, acquire the
    cutover locks, retire legacy nonterminal ledgers, switch all three new-run
    objectives, delete superseded pipelines, run static scans and docs checks,
    and record lifecycle closeout.

## Execution Model

- Parent agent owns plan state, architecture decisions, shared-contract
  integration, benchmark judgment, production cutover, and final sign-off.
- The parent first establishes the focused test contract in each checklist
  stage. One production-code subagent then owns only that stage's explicitly
  assigned production files; the parent owns test additions, integration,
  verification, evidence, lifecycle records, and public-model integration.
- After all planned implementation verification passes, one independent
  code-review subagent reviews the final integrated diff and does not edit.
  The parent remediates findings inside this plan's change surface and reruns
  affected gates.
- No subagent edits shared public models, production dispatcher, plan status,
  registry, or cutover code without an explicit parent-assigned boundary.
- Benchmark cases run sequentially because each real-LLM trace requires
  individual inspection.
- If native subagents are unavailable, stop before execution and request the
  user's explicit fallback authorization.

## Progress Checklist

- [x] Stage 1 - benchmark v2 contract and baseline frozen
  - Covers: steps 1-2.
  - Verify: `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_benchmark_contracts.py -q`; then run each locked baseline case with `venv\Scripts\python scripts\run_coding_agent_benchmark.py --engine-id pipeline_v1 --case-id <case_id>`.
  - Evidence: record manifest SHA-256, every result path, judgment note, and comparator precondition in `Execution Evidence`.
  - Handoff: production subagent starts Stage 2 only after the parent records the baseline.
  - Sign-off: Codex, 2026-07-11. Corrected-manifest baseline finalized with
    30 canonical results, preserved provisional inputs, trace-derived
    hard-gate/rubric judgments, and comparator precondition coverage.
- [x] Stage 2 - repository snapshot and overlay contracts
  - Covers: steps 3-4.
  - Verify: `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_repository_index.py tests\test_coding_agent_phase_d_candidate_recovery.py -q`.
  - Evidence: record focused failure before implementation, pass result, pinned-snapshot and recovery artifact paths.
  - Handoff: parent adds patch-operation tests for Stage 3.
  - Sign-off: Codex, 2026-07-11. The current focused suite proves the complete
    published snapshot schema, literal/regex/symbol/path search and cursor
    binding, streamed exclusion/resource handling, snapshot pin/reclamation,
    base/overlay merge, and replay-safe create/edit/delete/rename recovery.
- [x] Stage 3 - canonical patch operations and action-loop core
  - Covers: steps 5-6.
  - Verify: `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_patch_operations.py tests\test_coding_agent_phase_d_action_loop_contracts.py -q`.
  - Evidence: record failing then passing tests, operation/review artifacts, and prompt-render output.
  - Handoff: parent adds coding-run integration tests for Stage 4.
  - Sign-off: Codex, 2026-07-11. The current focused suite proves canonical
    create/edit/delete/rename records and candidate provenance, the strict
    action-specific protocol, durable action/observation recovery and budgets,
    and semantic run validation against trusted execution context.
- [x] Stage 4 - private evaluation lifecycle integration
  - Covers: steps 7-8.
  - Verify: `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_coding_run_integration.py tests\test_coding_agent_phase_d_action_loop_contracts.py tests\test_coding_agent_phase_d_patch_operations.py -q`.
  - Evidence: record isolated start/approval/resume handoff, engine isolation,
    candidate artifacts, and absence of public dispatch changes.
  - Handoff: parent starts real-LLM inspection at Stage 5.
  - Historical sign-off restored: Codex, 2026-07-11. The private evaluation
    start/approval/resume/blocker lifecycle reached its then-approved contract.
    Later cross-lifecycle findings are tracked in Stage 4R and do not alter
    this completed checkpoint.
- [x] Stage 4R - architecture closure and one-pass remediation
  - Origin: post-Stage-4 live evidence exposed cross-lifecycle failure modes
    that were not represented by the component-focused Stage 2-4 contracts.
  - Scope: create the complete failure-mode/provenance matrix; statically audit
    source-backed and source-free lifecycle transitions; close candidate,
    proposal, approval, apply-workspace, execution-workspace, blocker/resume,
    benchmark-scenario, and engine-freeze identities in one design; perform
    one cohesive implementation pass after the static review.
  - Test freeze: execute no deterministic test, benchmark, live-LLM case, or
    model call during this checkpoint.
  - Confidence gate: record repeated static assessments until confidence that
    the prepared harness will pass in one sequence is at least 90%.
  - Evidence: failure-mode matrix, ownership/provenance table, static call-path
    audit, one-pass change manifest, confidence assessments, and prepared but
    unexecuted harness sequence.
  - Handoff: request the user's explicit permission before executing any test;
    request separate explicit permission before executing real-LLM cases.
  - Sign-off: Codex, 2026-07-12. Static architecture closure, cohesive
    remediation, harness preparation, and Stage 5 handover are complete. The
    test freeze remained intact.
  - Static failure-mode matrix:

    | Area | Failure mode to close before the one-pass change | Deterministic owner |
    |---|---|---|
    | Task contract | Explicit deliverables or constraints disappear behind an empty acceptance-criteria field | Controller context projection preserves the literal goal and explicit structured criteria |
    | Action schema | Model emits operational fields that deterministic code ignores, or unbounded list/text/path values | Parser owns an exact minimal bounded schema; deterministic code owns resume targets |
    | Context budget | Changed paths, failures, finish/block fields, or one large observation exceed 50k or evict current failure evidence | Structural context reducer and prompt-safe observation projector |
    | Source/index | Run uses an unpinned, stale, incomplete, excluded, or mismatched base snapshot | Repository index identity, source lock, pin/release, and cursor binding |
    | Candidate mutation | An ordinary exception leaves a nonterminal journal after a terminal loop observation | Candidate transaction owner completes or rolls back before returning an ordinary error |
    | Candidate recovery | Crash phase cannot converge candidate, overlay, revision, and operation identity exactly once | Candidate journal and recovery gate under run/source locks |
    | Operation sequence | Source-free create/edit/delete/rename or multi-revision records compile from the wrong baseline | Canonical ordered candidate-view compiler |
    | Action persistence | Action exists without a terminal observation, or a nonmutating terminal action is silently discarded | Action/observation reconciler reconstructs or safely replays by action semantics |
    | Proposal | Review artifacts, candidate revision/tree, ordered operations, and proposal digest disagree | Proposal finalizer and canonical binding validator |
    | Approval | A structurally valid approval can authorize a different proposal or revision | Approval binding includes exact proposal digest and candidate revision |
    | Apply | Crash between approval, managed-copy apply, and state persistence repeats or loses the effect | Durable continuation-effect state persists before and after managed-copy apply |
    | Verification | Crash or repair reuses stale apply/candidate execution identity or loses the exact execution result | Durable verification attempt identity and current-candidate execution binding |
    | Repair | Successful current-candidate evidence leaves old failure marked unresolved, or old approval applies repaired content | Identity-bound failure status; repair finish creates a new proposal and approval requirement |
    | Block/resume | Model chooses deterministic routing fields, blocker answer is not persisted before a model call, or counters/history reset incorrectly | Deterministic blocker mapper and durable segment transition |
    | Concurrency | Evaluation setup bypasses run/source locks or two continuations interleave | Canonical async start/continue lock boundary |
    | Evaluation | Engine-neutral precondition has different candidate/journal operation identity across engines | Private evaluation scenario materializer with one canonical precondition identity |
    | Benchmark | Cohort mixes runtime closures, overwrites attempts, auto-scores behavior, or compares unequal scenario inputs | Engine/scenario digests, immutable attempt retirement, reviewed judgment binding, comparator |
    | Platform | Managed paths exceed Windows limits or newline/hash rules differ across boundaries | Short digest-based workspace names and one canonical content identity |
    | Privacy | Operational IDs, host paths, secret-like files, or raw execution internals enter model context | Safe source policy and model-facing evidence projector |

  - Confidence assessment 1, 2026-07-11: 64%. The component implementations
    are substantial, but static review found unresolved cross-lifecycle gaps in
    approval binding, apply/verification persistence, context projection,
    ordinary mutation rollback, nonmutating orphan handling, evaluation lock
    ownership, and failure-resolution state. This is below the 90% gate;
    production implementation remains frozen while static review continues.
  - Confidence assessment 2, 2026-07-11: 78%. The durable effect-state design
    now covers dispatch, mutation, proposal, approval, apply, verification,
    repair, blocker, and crash recovery. Additional static inspection found
    incomplete repository-build resume/reuse, pagination/overlay cursor drift,
    unsafe source-copy policy divergence, approval-provenance replay risk, and
    a model-facing blocker field owned by deterministic code. This remains
    below the 90% gate; production implementation and all harness execution
    remain frozen.

  - Static lifecycle transition contract:

    | Transition | Durable record written before the effect | Terminal evidence | Recovery rule |
    |---|---|---|---|
    | Controller action | Dispatch id derived from run id, sequence, and action id | Exactly one observation for the dispatch | Reconstruct read/search/note/finish/block from the persisted action semantics; recover edit/run from their own journals |
    | Candidate edit | Dispatch operation id, candidate revision, path precondition, and rollback backup | Committed or rolled-back journal row | Ordinary exceptions roll back before returning; only process interruption may leave an intermediate phase |
    | Proposal | Candidate revision, candidate tree digest, ordered operation records, and proposal digest | Immutable review artifact | Any later candidate edit invalidates the proposal and requires finalization again |
    | Approval | Approval provenance digest and source-message identity bound to the exact proposal digest, candidate revision, and candidate tree | Accepted or rejected approval-binding record | Replayed approval evidence or any binding mismatch is rejected without entering repair |
    | Apply | Pending effect with deterministic apply package id and complete approval/proposal binding | Persisted apply result | Retry the same deterministic managed-copy effect only when terminal evidence is absent |
    | Verification | Pending attempt identity containing apply package, execution-spec digest, and ordinal | Persisted execution result per attempt | Resume at the first attempt without terminal evidence; never reuse a prior candidate execution identity after repair |
    | Repair | Current verification failure identity and candidate revision | Resolved only by successful evidence for the current candidate | Candidate mutation clears prior approval/apply binding; finish creates a new proposal requiring a new approval |
    | Blocker response | User-answer observation and cleared blocker state | Persisted active segment state | Controller invocation begins only after the response transition is durable |

  - Provenance closure: `source snapshot -> candidate revision/tree -> ordered
    operations -> proposal digest -> approval binding -> deterministic apply
    package -> verification attempt/result`. Every arrow is validated by the
    deterministic owner before the next external effect. Operational ids and
    host paths remain in durable evidence and are projected out of model
    context.
  - Static benchmark contract audit, 2026-07-11:
    - `small_feature_cli` is invalid because its task requests an unnamed
      feature; the fixture only identifies ownership files.
    - `small_feature_counter` and `small_feature_slug` conflict with fixture
      language that protects or requires unchanged tests while their task asks
      the agent to add tests.
    - `revision_cli`, `preflight_cli`, `repair_cli`, and
      `concurrent_source_cli` do not name a requested runtime change.
    - the three `blocker_response` rows require `completed`, but the harness
      supplies no blocker answer or external dependency and therefore cannot
      exercise blocker continuation.
    - status equality alone is insufficient behavioral acceptance for the
      remaining categories. Harness preparation must replace ambiguous task
      text with literal fixture deliverables and add category-specific
      artifact/state assertions. Existing attempts remain immutable historical
      evidence and do not contribute to the post-remediation confidence gate.
  - Confidence assessment 3, 2026-07-11: 88%. The transition and provenance
    contracts now close the known production failure modes, and the benchmark
    audit separates invalid expectations from implementation failures. Static
    call-site review is still required for managed-copy policy, deterministic
    dispatch identity, and approval/apply recovery compatibility. This remains
    below the 90% gate, so the cohesive implementation pass has not started.
  - Confidence assessment 4, 2026-07-11: 92%. Static call-site review confirms
    the remediation can preserve existing public patch/application contracts:
    deterministic dispatch and apply-package identities are optional at the
    shared boundary and mandatory inside the action loop; public approval
    evidence remains unchanged while the loop creates the exact internal
    binding; the shared managed-copy policy replaces each coding-agent copy
    caller without changing destination layout. The one-pass implementation
    may now begin. Test execution remains frozen.

  - One-pass change manifest under static review:

    | Final owner | One-pass change |
    |---|---|
    | `code_action_loop/models.py` | Define the v2 durable dispatch, proposal, approval-binding, verification-state, and pending-effect shapes. |
    | `code_action_loop/context.py` | Project typed semantic observations; bound each observation, changed-path summary, failure state, criteria, and list field before final JSON rendering. |
    | `code_action_loop/parser.py` / `prompts.py` | Remove model-owned `resume_target`; bound every id/path/list/item/finish/block field; reject booleans where integers are required; require a literal-goal deliverable check before finish. |
    | `code_action_loop/state.py` | Guarantee every ordinary mutation return leaves a terminal journal; retain only crash-created nonterminal phases; expose operation-id recovery outcomes. |
    | `code_action_loop/actions.py` | Receive deterministic dispatch operation ids; enforce current candidate/revision identities; project execution evidence without operational ids or host details. |
    | `code_action_loop/supervisor.py` | Persist dispatch before effects; reconcile by action semantics; bind approval provenance to exact proposal/revision/tree; durably advance apply/verification effects; reset segments through one owner; track current verification state instead of an append-only unresolved string list. |
    | `code_patching/models.py` / `apply.py` | Carry and validate internal approval binding with canonical proposal/revision; distinguish authorization rejection from repairable apply failure; publish managed-copy evidence before verification. |
    | `code_executing/models.py` / `supervisor.py` | Keep current-candidate/apply execution identities outside model evidence; persist exact results for crash reconciliation; retain short Windows-safe workspace names. |
    | `repository_index/builder.py` / `storage.py` | Make `.building` file progress idempotent, eliminate duplicate FTS/symbol/import rows, and perform actual unchanged-content row reuse from complete snapshots. |
    | `repository_index/search.py` / `overlay.py` | Page the merged candidate view with one cursor identity; apply glob before page completion; prevent repeated overlay rows and empty unchanged cursors; align base/overlay literal and regex semantics. |
    | `safety.py` and managed-copy callers | Use one non-symlink, secret/environment-excluding source-copy policy for candidate, apply, and execution workspaces. |
    | `coding_run/evaluation.py` | Materialize engine-neutral scenarios under canonical locks and preserve one operation id across precondition, journal, review, approval, and apply. |
    | benchmark runner | Preserve immutable attempts, bind scenario/engine closure, prepare sequential reviewed cases, and remain unexecuted until permission. |
- [ ] Stage 5 - real-LLM, benchmark, and post-gate public cutover
  - Covers: steps 9-10.
  - Verify: run one real-LLM case at a time; run every action-loop benchmark case individually; run the comparator and post-cutover scans below.
  - Evidence: record each trace/judgment, comparison artifact, public cutover,
    review findings/remediation, locks, and scan results.
  - Handoff: archive only after all gates pass.
  - Status: paused by user on 2026-07-11 while Stage 4R performs architecture
    closure. Preserve all reviewed, provisional, retired, and raw artifacts;
    execute no additional case until the user authorizes test execution.
  - Sign-off: `<agent/date>` after verification and evidence; reread this plan before final lifecycle update.

## Historical Execution Handover - Stage 4R Implementation

### Handover Purpose And Authority

- Disposition: consumed by the signed Stage 4R execution on 2026-07-12. The
  Stage 5 handover under `Execution Evidence` is now authoritative.

- The next implementation agent owns Stage 4R execution under this
  `in_progress` plan. The agent may reconcile and implement the Phase D
  production-code scope named below.
- The current parent architecture task ends with this handover. The next agent
  becomes responsible for implementation decisions already fixed by this
  section, progress reporting, static confidence assessment, and preparation
  of the unexecuted verification harness.
- Test execution remains frozen. The agent must request the user's explicit
  permission before running any deterministic test, static test runner,
  benchmark, model smoke, or live-LLM case. Real-LLM execution requires a
  separate explicit permission after deterministic verification is approved.
- Completed checkpoints are immutable. Never uncheck, reopen, replace, or
  erase Stage 1, Stage 2, Stage 3, or Stage 4. Record later findings only under
  Stage 4R and append evidence rather than rewriting historical sign-off.
- Reread this entire plan after every future major-stage sign-off and before
  starting the next stage.

### Current Execution State

- The previous execution subagent is halted. No execution subagent is active.
- Stage 4R is the next unchecked checkpoint. Stage 5 remains paused.
- The test freeze has remained intact since the user imposed it.
- The working tree contains interrupted Phase D implementation, test, plan,
  and benchmark changes. It also contains later unverified draft edits made
  during architecture review. Treat the entire dirty Phase D diff as candidate
  material requiring reconciliation; file presence, changed lines, and staged
  state are not completion evidence.
- Preserve the shared working tree. Do not use blanket reset, checkout,
  recursive deletion, or broad reversion. Reconcile the named Phase D files in
  place against the fixed contracts below, preserving valid prior work and
  replacing only conflicting draft logic.
- The 92% confidence recorded in Confidence Assessment 4 is architecture and
  change-surface readiness. It is not implementation confidence and cannot be
  used to authorize the harness. Record a new post-implementation confidence
  assessment after the cohesive code pass.
- Existing reviewed benchmark artifacts remain immutable historical evidence.
  Current source changes and manifest edits create a new future engine/scenario
  closure and must never overwrite or relabel earlier attempts.

### Fixed Architectural Decision

The implementation agent must implement this architecture directly and must
not introduce compatibility shims, parallel action vocabularies, fallback to
the manager/programmer exploration pipeline, or test-shaped prompt rules.

1. **One generic semantic action loop**
   - Use one strict JSON action protocol for `read`, `search`, `edit`, `run`,
     `note`, `finish`, and `block`.
   - Use the same loop for source-backed and source-free work. Candidate state,
     rather than source origin, owns the mutable repository view.
   - The controller chooses one semantic action per turn. It never supplies a
     command, workspace path, approval decision, resume route, lock policy,
     persistence policy, or retry policy.
   - Deterministic code owns schema validation, safe paths, permissions,
     locking, limits, cursor binding, candidate mutation, persistence,
     recovery, approval, apply, execution, and blocker routing.

2. **Local-LLM model boundary**
   - Keep the stable action vocabulary and output schema in the system prompt.
     Keep the current goal and projected observations in the dynamic message.
   - Project semantic evidence rather than raw operational data. The model may
     receive bounded source excerpts, paths, symbols, candidate revision,
     semantic execution outcome, current failure, and explicit deliverables.
   - Keep host paths, operation ids, package ids, snapshot hashes, approval
     provenance internals, execution identity, lock identity, and raw workspace
     references outside model context.
   - Remove `resume_target` from the model-owned block schema. The deterministic
     blocker mapper derives the resume target from the blocker type.
   - Preserve the literal user goal and explicit acceptance criteria. Before
     `finish`, require the controller to account for each explicitly requested
     runtime artifact, test artifact, deletion, rename, constraint, and known
     verification limitation. Ask for scope clarification when the requested
     behavior is unnamed.

3. **Closed provenance chain**
   - Enforce this exact chain:

     `source snapshot -> candidate revision/tree -> ordered operations -> proposal digest -> approval binding -> deterministic apply effect -> verification attempt/result`

   - Validate every arrow before the next external effect.
   - Derive an operation id from run id, action sequence, and action id before
     dispatch; persist it with the action before mutation or execution.
   - Finalization binds candidate revision, candidate tree digest, ordered
     canonical operations, review artifacts, and proposal digest.
   - Internal approval binding combines the exact proposal digest, candidate
     revision, candidate tree digest, approval-evidence digest, and approval
     source-message identity. A different or previously consumed approval
     identity cannot authorize another proposal.
   - Any candidate edit after finalization invalidates the proposal, approval,
     apply effect, and verification binding. A repaired candidate requires new
     finalization and new approval.

4. **Durable effect and crash semantics**
   - Persist an action dispatch before its deterministic effect and persist one
     terminal observation before the next model call.
   - Ordinary candidate-mutation errors finish as `committed` or `rolled_back`.
     Only process interruption may leave `prepared`, `candidate_written`, or
     `overlay_written` for recovery.
   - Reconcile orphan actions by action semantics. Recover edits from the
     candidate journal, runs from exact execution evidence, and safely replay
     deterministic non-mutating actions. Reconstruct `finish` and `block`
     terminal effects instead of discarding them.
   - Persist blocker-response observation and the cleared blocker state before
     invoking the controller again.
   - Persist approval plus the complete pending apply/verification effect before
     creating a managed apply workspace. Use a deterministic apply-package id
     so a missing terminal result can resume without creating a second effect.
   - Persist each verification-attempt identity before execution and its exact
     result immediately afterward. A crash with an active attempt rematerializes
     a clean deterministic apply workspace and restarts the saved verification
     sequence from its defined recovery point.
   - Track the current unresolved failure by candidate/attempt identity. A
     successful current-candidate run resolves it; append-only historical
     observations remain available for audit.

5. **Persistent repository index and uncapped exploration**
   - Publish immutable SQLite snapshots under the stable source identity and
     pin the selected snapshot for every live run.
   - Make `.building` progress file-transactional and idempotent. Resume without
     duplicate `file`, `chunk`, FTS, `symbol`, or `import_edge` rows.
   - Reuse complete unchanged file rows from a compatible published snapshot
     instead of rereading and reparsing them.
   - Merge base snapshot and candidate overlay into one candidate-view search
     order. Bind cursors to snapshot, overlay revision, mode, query, glob, and
     last sort position. Apply glob filtering before page completion, prevent
     repeated overlay rows, and guarantee cursor progress or terminal `None`.
   - Align literal, regex, symbol, and path semantics between base and overlay;
     return match-centered bounded excerpts for literal and regex results.
   - Remove fixed file, repository-map, wave, programmer, report-count, and
     controller-turn exploration caps. Retain deterministic wall-time,
     persistence/storage, output-size, execution-count, and no-progress safety
     budgets because these constrain resources rather than repository breadth.

6. **Canonical create/edit/delete/rename and managed copies**
   - Compile ordered operations against the current candidate view. Preserve
     operation order across create, anchor edit, small-file replace, delete,
     and rename.
   - Review and approval artifacts must expose delete and rename explicitly and
     bind both rename paths.
   - Use one shared managed-copy policy for candidate, review/preflight, apply,
     and execution workspaces. Exclude symlinks, `.env` variants, secret-like
     paths, VCS metadata, caches, and coding-agent generated workspace roots.
   - Keep original resolved source immutable. Apply and execution occur only in
     managed copies.

7. **Private evaluation and benchmark boundary**
   - Keep `action_loop_v1` private until Stage 5 cutover. The public engine must
     not switch per run and must not fall back after a private-engine failure.
   - Materialize engine-neutral benchmark preconditions through the canonical
     run/source lock boundary and use one operation identity across scenario,
     journal, review, approval, apply, and comparison artifacts.
   - Benchmark judgments remain human/reviewer-authored. Deterministic code may
     validate artifact closure and calculate aggregates; it cannot manufacture
     behavioral scores from final status.

### Allowed Change Surface

The implementation agent may reconcile these Phase D owners:

- `src/kazusa_ai_chatbot/coding_agent/code_action_loop/`
- `src/kazusa_ai_chatbot/coding_agent/repository_index/`
- `src/kazusa_ai_chatbot/coding_agent/safety.py`
- `src/kazusa_ai_chatbot/coding_agent/context_budget.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/models.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_operations.py`
- `src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py`
- managed-copy integration in
  `src/kazusa_ai_chatbot/coding_agent/code_patching/patch_validation.py`
- current-candidate identity and managed-copy integration in
  `src/kazusa_ai_chatbot/coding_agent/code_executing/models.py` and
  `src/kazusa_ai_chatbot/coding_agent/code_executing/supervisor.py`
- private-engine materialization in
  `src/kazusa_ai_chatbot/coding_agent/coding_run/evaluation.py`
- Phase D configuration and route-closure fields in
  `src/kazusa_ai_chatbot/config.py` and
  `src/kazusa_ai_chatbot/llm_interface/route_report.py`
- Phase D README/ICD documentation named by the existing plan.

The following remain outside Stage 4R production scope:

- public cutover and deletion of the legacy manager/programmer pipeline;
- unrelated coding-agent phases or chatbot brain modules;
- package installation, environment mutation, or `.env` inspection;
- compatibility aliases, dual-write state, runtime feature flags, or fallback
  engines;
- production changes made solely to satisfy an ambiguous benchmark task.

### Known Draft-Diff Risks To Resolve

The implementation agent must inspect the current dirty diff for these exact
risks before treating any draft code as complete:

| Risk | Required closure |
|---|---|
| Parser and prompt drift | Parser, prompt, direct controller fixtures, and deterministic blocker mapper share one four-field model-owned block schema; only deterministic state contains `resume_target`. |
| Context overexposure or eviction | Search excerpts remain visible, operational identities remain private, and the latest current failure survives the final 50k render bound. |
| Candidate rollback gaps | Every caught ordinary edit failure leaves a terminal journal and a later action can proceed; crash recovery returns an operation-specific outcome. |
| Approval crash window | Approval binding and pending effect become durable in one state transition before apply; recovery does not require consuming a second user approval. |
| Apply replay drift | Deterministic package recreation is confined to its managed package path and validates the same approval/proposal/tree binding. |
| Verification replay drift | An interrupted active attempt resumes from a clean candidate/apply identity and does not skip or double-count results. |
| Orphan terminal actions | `finish` and `block` recover their state transition, working note, finalization, and public projection. |
| Index resume duplication | Per-file transactions clean partial rows, including FTS rows, before replay and mark completion only after all metadata is durable. |
| Cross-snapshot reuse | Reused rows come only from a complete compatible snapshot with the same normalized content hash and preserve current snapshot/file identities. |
| Candidate-view pagination | Glob-filtered empty pages still advance; terminal pages return no cursor; overlay rows do not repeat across pages. |
| Managed-copy identity | Shared exclusions cannot cause candidate-tree and execution-tree digests to disagree. |
| Evaluation lock ownership | Private scenario setup never calls the unlocked preparation helper outside the canonical run/source lock scope. |
| Benchmark validity | Ambiguous feature cases and blocker-response expectations are corrected in the harness rather than encoded as prompt special cases. Historical attempts keep their original manifest/engine digests. |

### Required Execution Sequence

The implementation agent must follow this sequence without interleaving test
execution:

1. **Inventory and reconcile the dirty Phase D diff.**
   - Reread this plan, `development_plans/README.md`, the mandatory skills,
     subsystem READMEs, relevant source, relevant tests, and `git status --short`.
   - Map every changed Phase D file to the fixed architecture and the Stage 4R
     failure-mode matrix.
   - Record which draft hunks are retained, rewritten, or deferred. Preserve
     unrelated user changes.

2. **Freeze the complete contracts before further code edits.**
   - Confirm the action schema, context projection, durable state fields,
     provenance chain, recovery transitions, repository cursor, approval
     binding, pending effect, and private evaluation identity as one closed
     call graph.
   - Confirm each model-facing field answers a semantic controller need and
     each operational field remains deterministic.

3. **Perform one cohesive implementation pass.**
   - Reconcile model/parser/prompt/context first.
   - Reconcile candidate journal, deterministic action dispatch, supervisor
     transitions, approval/apply/verification persistence, and blocker resume
     second.
   - Reconcile persistent index build/reuse, merged pagination, and managed-copy
     policy third.
   - Reconcile private evaluation ownership and documentation last.
   - Keep edits within the allowed change surface and implement the fixed
     contracts directly.

4. **Perform static failure-mode review.**
   - Trace every transition in the Stage 4R matrix through its writer, durable
     artifact, recovery reader, and next consumer.
   - Inspect all call sites for changed signatures and state keys.
   - Inspect the final diff for duplicate state vocabularies, host/context
     leaks, unsafe copy behavior, stale proposal reuse, nonterminal ordinary
     failures, and fixed exploration caps.
   - This is source inspection only while the test freeze is active.

5. **Record post-implementation confidence.**
   - Score confidence that the complete deterministic gate will pass in one
     sequence and state the evidence for the score.
   - When confidence is below 90%, return to contract and call-path inspection,
     make one static remediation pass, and reassess. Repeat until confidence is
     at least 90%.
   - Architecture Confidence Assessment 4 cannot substitute for this new
     implementation-confidence record.

6. **Prepare the harness without executing it.**
   - Align existing Phase D deterministic fixtures with the final schema and
     transition contracts. Add only checks mapped to the known failure-mode
     table.
   - Correct the already identified ambiguous benchmark tasks and the
     blocker-response lifecycle in the manifest/runner. Preserve historical
     attempts and produce a new manifest digest for future runs.
   - Prepare the exact deterministic command and the ordered one-case-at-a-time
     real-LLM command list. Do not invoke either command.

7. **Hand back for permission.**
   - Report changed files, static review findings, final implementation
     confidence, prepared deterministic command, prepared real-LLM sequence,
     and residual risks.
   - Wait for explicit deterministic-test permission. After deterministic
     results are reviewed, wait for separate explicit real-LLM permission.

### Prepared Verification Order

Preparation means commands and fixtures are ready; execution remains gated by
the user.

1. Focused deterministic modules, in dependency order:
   - `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_repository_index.py tests\test_coding_agent_phase_d_candidate_recovery.py -q`
   - `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_patch_operations.py tests\test_coding_agent_phase_d_action_loop_contracts.py -q`
   - `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_coding_run_integration.py -q`
   - `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_benchmark_contracts.py -q`
2. Complete deterministic Phase D gate:
   - `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_repository_index.py tests\test_coding_agent_phase_d_candidate_recovery.py tests\test_coding_agent_phase_d_patch_operations.py tests\test_coding_agent_phase_d_action_loop_contracts.py tests\test_coding_agent_phase_d_coding_run_integration.py tests\test_coding_agent_phase_d_benchmark_contracts.py -q`
3. Real-LLM controller smoke cases:
   - Run only the named cases in
     `tests/test_coding_agent_phase_d_action_loop_live_llm.py`, one pytest node
     at a time, inspecting and recording each trace before the next node.
4. Action-loop benchmark:
   - Run `venv\Scripts\python scripts\run_coding_agent_benchmark.py --engine-id action_loop_v1 --case-id <case_id>` once per prepared manifest row, in manifest order.
   - Review and bind one judgment artifact before advancing to the next case.
5. Comparator, public cutover, legacy removal, and independent code review:
   - These remain Stage 5 work after all earlier gates pass and the user
     authorizes the cutover sequence.

### Stage 4R Handover Completion Criteria

Stage 4R may be checked and signed only when all of the following evidence is
appended under `Execution Evidence`:

- dirty-diff reconciliation record;
- final changed-file list within the allowed surface;
- closed state/provenance call-path review;
- static failure-mode review results;
- post-implementation confidence of at least 90% with rationale;
- prepared deterministic and real-LLM harness commands;
- confirmation that the test freeze remained intact;
- explicit pointer to Stage 5 as the next unchecked checkpoint;
- agent/date sign-off.

The implementation agent must leave Stage 4R unchecked when any item is
missing. Stage 5 remains unchecked until its own verification, reviewed
evidence, cutover, and independent code-review gates complete.

## Verification

### Deterministic Contract Verification

- Before each implementation stage, run its named focused test from the
  checklist and record the expected missing-symbol or assertion failure. After
  implementation, the same command must pass with exit code 0.
- The complete deterministic gate is:
  `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_repository_index.py tests\test_coding_agent_phase_d_candidate_recovery.py tests\test_coding_agent_phase_d_patch_operations.py tests\test_coding_agent_phase_d_action_loop_contracts.py tests\test_coding_agent_phase_d_coding_run_integration.py tests\test_coding_agent_phase_d_benchmark_contracts.py -q`.
  It must exit 0 with no skipped target-state contract tests.

- Strictly parse every action variant and reject malformed JSON, unknown keys,
  wrong action capabilities, stale revisions, invalid paths, and invalid
  operation-specific fields.
- Prove invalid outputs become bounded observations and three consecutive
  invalid outputs become the specified typed blocker.
- Prove action events and observations are durable, ordered, resumable, and
  mutation replay-safe after simulated interruption.
- Prove run-level budgets stop with typed blockers and do not silently truncate
  a successful result.
- Prove the context reducer preserves mandatory current state and evicts oldest
  resolved observations first.
- Prove source manifests and snapshot ids change when content, exclusion policy,
  or index schema changes.
- Prove incomplete index builds are not queryable and resume to one atomically
  complete snapshot.
- Prove unchanged file hashes reuse indexed content without stale path, symbol,
  or excerpt results.
- Prove a repository with more than 120 safe files indexes all safe files and a
  search can discover a target beyond the former cap without pre-supplying its
  path.
- Prove literal, regex, symbol, and path modes have deterministic ordering,
  cursor binding, prompt-safe excerpts, and stale-cursor rejection.
- Prove create/replace/insert results appear immediately through the overlay.
- Prove delete tombstones hide base results and rename exposes only the target
  path with content identity preserved.
- Prove delete/rename reject symlinks, binaries, directories, protected paths,
  missing/stale hashes, existing targets, case collisions, out-of-scope paths,
  and cross-source moves.
- Prove review packages and final application represent and bind delete/rename
  exactly, while original source remains unchanged before explicit apply.
- Prove semantic run requests cannot carry commands and cannot bypass Phase B
  availability, approval, revision, or execution-plan validation.
- Prove all three public coding objectives reach the same loop engine with the
  correct capability set.

### Patched LLM Handoff Verification

- Run `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_action_loop_contracts.py tests\test_coding_agent_phase_d_coding_run_integration.py -q` with patched model output fixtures. It must exit 0 and persist the asserted action, observation, and context-manifest artifacts under the test workspace.

- Patch only model responses and verify coding-run-to-loop state handoff,
  action dispatch, observation merge, finish projection, typed blocker
  projection, approval continuation, and verification-failure repair resume.
- Verify a read-only run cannot dispatch edit or run even if the patched model
  asks for them.
- Verify proposal/repair run availability changes are supplied semantically and
  invalid stale actions are rejected by deterministic state.
- Verify no handoff imports or routes through removed manager/programmer
  packages.

### Real LLM Verification

- Create 2-5 individually inspectable cases for the controller prompt covering:
  targeted search/read, multi-file exploration, create/replace, delete, rename,
  run-result interpretation, failure repair, and a legitimate blocker.
- Create at least one full real-LLM graph case for each objective:
  `read_only`, `propose_patch`, and `verify_repair`.
- Each case must persist input, model route/config, prompt version, raw output,
  parsed action, action/observation trace, context manifests, final result, and
  human/agent judgment notes.
- Hard gates cover schema, action vocabulary, path/source safety, approval
  boundaries, source-grounded evidence refs, and absence of secret/host-path or
  unavailable-action claims.
- Behavioral rubrics cover task progress, useful next-action choice,
  repository-grounded editing, response to execution evidence, repair quality,
  and knowing when to finish or block.
- Run and inspect one case at a time. A pytest pass without trace review is not
  acceptance evidence.
- Invoke each case with `venv\Scripts\python -m pytest <target> -q -s -m live_llm`; record its exact test target, route/config digest, trace path, and judgment. A missing trace, missing parsed action, or missing human/agent judgment fails the case.

### Benchmark Cutover Gate

- Use Phase C's frozen versioned 30-case manifest and run both engines against
  identical source fixtures, task inputs, model route/config, and scoring
  rubric.
- Run real-LLM cases one at a time and retain separate durable artifacts for
  `pipeline_v1` and `action_loop_v1`.
- Required cutover thresholds:
  - `action_loop_v1` end-state success rate is at least 10 percentage points
    above the frozen `pipeline_v1` baseline;
  - no category has a lower safety/privacy/authorization score;
  - zero original-source mutation before approved apply;
  - zero stale-index accepted results;
  - zero command-policy or approval bypasses;
  - all repository-scale, delete, and rename cases pass their hard gates;
  - median model calls and wall time are reported, not used to waive quality or
    safety gates.
- A threshold failure blocks production cutover and produces a scoped
  remediation entry in Execution Evidence.

### Post-Cutover Scans

- `rg -n 'code_reading|code_writing|code_modifying|repository_map|MAX_(REPOSITORY_MAP_FILES|PROGRAMMERS_PER_WAVE|PROGRAMMER_WAVES|PROGRAMMER_REPORTS_PER_PM|FILES_PER_PROGRAMMER|WRITE_FALLBACK_EVIDENCE_FILES)' src/kazusa_ai_chatbot/coding_agent` returns no matches. Exit code 1 is the expected successful zero-match result; any match blocks completion unless it is a documented archived artifact outside `src`.
- `rg -n 'pipeline_v1|tool_call|function_call|create_child_pm|create_programmer_task' src/kazusa_ai_chatbot/coding_agent` returns no production match. Exit code 1 is expected; any match blocks completion.
- `rg -n 'CODING_AGENT_PROGRAMMER_LLM|extract_additive_execution_specs|ADDITIVE_EXECUTION_SPEC_PROMPT|MAX_ADDITIVE_SAFE_TEST_PATHS' src/kazusa_ai_chatbot` returns no match. Exit code 1 is expected; any match blocks completion.
- `rg -l 'CODING_AGENT_PM_LLM|LLInterface|LLMCallConfig' src/kazusa_ai_chatbot/coding_agent` returns only `code_fetching/source_intake.py` and `code_action_loop/supervisor.py`; the latter must contain `CODING_AGENT_ACTION_LOOP_LLM` and must not contain `CODING_AGENT_PM_LLM`. Any additional coding-agent LLM caller blocks completion.
- `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_coding_run_integration.py -q` proves exactly one action-loop route per objective and no runtime engine switch.
- Review `src/kazusa_ai_chatbot/coding_agent/README.md`, the listed subsystem READMEs, and `docs/SUBAGENT_INTERFACES.md` against the integrated code. Record each reviewed file and confirm it names the final public/run boundary only.

## Independent Code Review

- Reviewer must be independent from the primary implementer and must read this
  full plan before reviewing.
- Review the final integrated diff, not isolated child-agent patches.
- Required focus:
  - public coding-run ownership and big-bang cutover completeness;
  - index identity, invalidation, exclusion policy, atomic publication, and
    stale-result prevention;
  - overlay/base merge correctness;
  - strict JSON parsing, validation, recovery, and no-progress behavior;
  - delete/rename path, hash, collision, digest, review, and apply safety;
  - execution/approval/revision binding;
  - original-source immutability;
  - local-LLM context size and prompt-safe observations;
  - removal of fixed exploration caps and superseded runtime paths;
  - benchmark evidence integrity and one-at-a-time trace judgment.
- Critical and high findings block completion and must be remediated and
  re-reviewed.
- Medium findings require remediation or an explicit user-approved deferral
  recorded in Execution Evidence.

## Acceptance Criteria

- `read_only`, `propose_patch`, and `verify_repair` all execute through one
  `coding_action.v1` controller and durable loop state.
- The model can iteratively read and search any safe indexed file without fixed
  repository, wave, agent, report, or file-count caps.
- Repository indexing persists across runs, reuses unchanged content, pins a
  complete snapshot, excludes unsafe content, and never returns stale base or
  candidate results as current.
- Candidate search reflects every successful edit at the same candidate
  revision.
- Delete and rename are safe, source-scoped, preconditioned, candidate-only
  before apply, visible in review, and bound to proposal/approval/execution
  provenance.
- Run actions remain semantic and can invoke only Phase B's deterministic
  allowlisted execution boundary.
- Execution failures and finalization failures return to the same loop with
  structured evidence and a bounded repair budget.
- Budget exhaustion, invalid-controller repetition, and unresolved external
  needs become typed Phase C blockers with exact resume targets.
- The production cutover passes all deterministic/handoff verification, the
  real-LLM cases, the benchmark thresholds, independent review, and
  post-cutover scans.
- Production contains no runtime fallback, compatibility alias, fixed
  exploration topology, or manager/programmer model pipeline for the replaced
  objectives.
- Current architecture-shaped tests do not survive solely to preserve removed
  behavior; replacement tests assert the target contracts in this plan.

## Execution Evidence

- Status: in progress.
- Stage 4R static implementation sign-off, Codex, 2026-07-12:
  - **Execution boundary:** Stage 4R is complete. Stage 5 is the next unchecked
    checkpoint. No Stage 5 test, benchmark, model, comparator, cutover, legacy
    removal, or independent code-review action was executed.
  - **Test-freeze evidence:** the Stage 4R pass used source/document reads,
    `rg`, `git status --short`, `git diff`, `git diff --check`, line-length
    inspection, and `Get-FileHash`. It executed no pytest command, test
    harness, benchmark case, model smoke, live-LLM request, compilation check,
    or model call. `git diff --check HEAD` completed without whitespace errors;
    its only output was the repository's existing LF-to-CRLF warnings.
  - **Dirty-diff reconciliation:** every dirty file was reconciled against the
    fixed Stage 4R architecture. The generic action protocol, persistent index,
    canonical operation compiler, managed candidate/apply/execution copies,
    private evaluation engine, and paired benchmark structure were retained.
    Conflicting or incomplete draft logic was rewritten where it created a
    second action path, incomplete model redaction, non-identical replay,
    terminal-recovery drift, unbounded candidate regex execution, unsafe
    symlink traversal, invalid snapshot reuse, or an unloadable harness. Public
    cutover and legacy-pipeline removal remain deferred to Stage 5 as planned.
  - **Final changed-file inventory:**
    - lifecycle records: `development_plans/README.md` and this plan;
    - private benchmark: `scripts/run_coding_agent_benchmark.py` and
      `tests/fixtures/coding_agent_benchmark/cases.jsonl`;
    - generic loop: all nine files under
      `src/kazusa_ai_chatbot/coding_agent/code_action_loop/`;
    - repository index: all nine files under
      `src/kazusa_ai_chatbot/coding_agent/repository_index/`;
    - shared/runtime integration:
      `src/kazusa_ai_chatbot/coding_agent/safety.py`,
      `src/kazusa_ai_chatbot/coding_agent/context_budget.py`,
      `src/kazusa_ai_chatbot/config.py`, and
      `src/kazusa_ai_chatbot/llm_interface/route_report.py`;
    - patching integration:
      `code_patching/README.md`, `apply.py`, `models.py`,
      `patch_operations.py`, and `patch_validation.py`;
    - execution integration:
      `code_executing/README.md`, `models.py`, and `supervisor.py`;
    - private run integration: `coding_run/README.md` and
      `coding_run/evaluation.py`;
    - prepared deterministic/live harness:
      `tests/test_coding_agent_benchmark_contracts.py`,
      `tests/test_coding_agent_phase6_code_executing_contracts.py`,
      `tests/test_coding_agent_phase_d_action_loop_contracts.py`,
      `tests/test_coding_agent_phase_d_action_loop_live_llm.py`,
      `tests/test_coding_agent_phase_d_benchmark_contracts.py`,
      `tests/test_coding_agent_phase_d_candidate_recovery.py`,
      `tests/test_coding_agent_phase_d_coding_run_integration.py`,
      `tests/test_coding_agent_phase_d_patch_operations.py`,
      `tests/test_coding_agent_phase_d_repository_index.py`, and
      `tests/test_llm_interface_route_report.py`.
  - **Implemented Stage 4R corrections:**
    - model-facing working notes, source excerpts, execution excerpts,
      limitations, and patch-operation strings now share absolute-host-path
      redaction while the literal goal and criteria remain intact;
    - one shared managed-path predicate now rejects unsafe relative paths,
      symlinked managed roots, symlinked affected paths, and symlinked existing
      ancestors before candidate, overlay, recovery, patch compilation, apply,
      execution, index, or pin content is read or changed;
    - candidate state loading validates revision and journal structure;
      committed operation-id replay now requires the exact original operation,
      paths, revisions, source hash, and result hash;
    - recovery backup reads, writes, cleanup, rollback, and overlay rebuild are
      confined to the candidate recovery root, and normalized text identity is
      derived from the already validated bytes;
    - orphan reconciliation now considers only the latest observed action for
      terminal recovery. Observed or orphaned finish finalization failures
      persist the same typed current failure and failure observation as the
      uninterrupted path, then continue the controller loop. Reconciled state
      is persisted before the next controller invocation;
    - the alternate test-only `run_recorded_actions` persistence path and its
      topology-owned tests were removed, leaving the canonical supervisor as
      the only generic action-loop owner;
    - fresh loop state now defines approval, apply-attempt,
      execution-attempt, and effect-history collections at construction;
      pending-effect recovery validates schema and active state before replay;
    - published index reuse validates the exact source, manifest, schema,
      exclusion-policy, snapshot, and complete-state identity. Index, snapshot,
      pin, and overlay symlinks fail closed;
    - candidate overlay regex uses a bounded worker process and produces no
      partial timeout page, matching the immutable-snapshot regex boundary;
      overlay output remains page-resource bounded without restoring a
      repository exploration cap;
    - deterministic apply and execution package paths reject symlink
      redirection before package replacement or command execution;
    - the benchmark harness no longer sends model-owned `resume_target`, the
      ambiguous counter/slug/release-feed tasks name their exact deliverables,
      and a duplicate `trace_paths` keyword that prevented the runner source
      from loading was removed;
    - prepared contract cases now cover operation replay mismatch, host-path
      redaction across notes/evidence, and rejection of an older observed
      finish as the current terminal action.
  - **Closed state/provenance call-path review:**
    - `source snapshot -> pinned snapshot id`: builder publishes only an exact
      complete identity; the run pins it under the source/run lock;
    - `pinned snapshot -> candidate revision/tree`: candidate copy and overlay
      share the managed-copy and confined-path policy; every mutation is
      journaled before its filesystem effect;
    - `candidate revision/tree -> ordered operations`: committed edit evidence
      is merged once, mutation invalidates prior proposal/effect fields, and
      finalization compiles the ordered current-candidate operations;
    - `ordered operations -> proposal digest`: canonical records bind operation
      id, kind, both rename paths, source/result hashes, and candidate revision;
    - `proposal -> approval binding`: approval evidence digest and source
      message identity are consumed once and bound to the exact proposal,
      revision, and tree;
    - `approval -> apply effect`: approval and complete pending effect are
      persisted together before the deterministic managed apply package;
    - `apply -> verification attempt/result`: the apply tree is checked against
      the reviewed tree, attempt identity is persisted before execution, and
      exact terminal evidence is persisted before advancing;
    - `verification failure -> repair`: current failure is bound to candidate
      and effect identity; a repair edit clears stale proposal/approval/effect
      state and requires fresh finalization and approval;
    - `blocker response -> resume`: user-answer observation and cleared blocker
      state are durable before deterministic routing invokes the controller.
  - **Static failure-mode result:** no known critical or high Stage 4R finding
    remains open after the second source pass. The remaining risks are the
    intentionally unexecuted deterministic/runtime behavior, local-model
    behavior, and public cutover/legacy removal. Those are Stage 5 gates rather
    than waived Stage 4R findings.
  - **Post-implementation confidence assessment 5:** 84%. The first complete
    source pass found incomplete evidence redaction, replay identity drift,
    stale terminal selection, recovery finalization drift, symlink redirection,
    unbounded overlay regex evaluation, a model-owned blocker field in the
    harness, ambiguous tasks, and a duplicate runner keyword. This was below
    the 90% gate, so the implementation returned to the call-path review.
  - **Post-implementation confidence assessment 6:** 92%. The consolidated
    remediation above is now reflected at every writer/recovery-reader/consumer
    boundary; changed signatures have one call site; the alternate action path
    is absent; fixed exploration topology is absent from the private loop; the
    prepared harness matches the four-field model blocker schema; line-length
    inspection found no over-88 production lines; and `git diff --check HEAD`
    is clean. The remaining 8% represents evidence obtainable only through the
    frozen deterministic and live gates.
  - **Prepared deterministic commands, unexecuted, in dependency order:**
    1. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_repository_index.py tests\test_coding_agent_phase_d_candidate_recovery.py -q`
    2. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_patch_operations.py tests\test_coding_agent_phase_d_action_loop_contracts.py -q`
    3. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_coding_run_integration.py -q`
    4. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_benchmark_contracts.py -q`
    5. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_repository_index.py tests\test_coding_agent_phase_d_candidate_recovery.py tests\test_coding_agent_phase_d_patch_operations.py tests\test_coding_agent_phase_d_action_loop_contracts.py tests\test_coding_agent_phase_d_coding_run_integration.py tests\test_coding_agent_phase_d_benchmark_contracts.py -q`
  - **Prepared real-LLM commands, unexecuted and separately permissioned:**
    1. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_action_loop_live_llm.py::test_controller_live_chooses_targeted_repository_search -q`
    2. inspect and record the first trace before
       `venv\Scripts\python -m pytest tests\test_coding_agent_phase_d_action_loop_live_llm.py::test_controller_live_changes_request_after_empty_path_search -q`;
    3. inspect and record the second trace before any benchmark case.
  - **Prepared benchmark sequence, unexecuted:** the future manifest SHA-256 is
    `c5ed3cfc00dad7697366a2ec7963f80fc34b4458dcc469ab730d0a30bbb37bbf`.
    Run exactly one command of the form
    `venv\Scripts\python scripts\run_coding_agent_benchmark.py --engine-id action_loop_v1 --case-id <case_id>`
    and bind its reviewed judgment before the next case, in this order:
    `source_backed_preflight_bugfix`, `source_backed_cli_bugfix`,
    `source_backed_slug_bugfix`, `small_feature_cli`,
    `small_feature_counter`, `small_feature_slug`,
    `mixed_create_edit_approval`, `mixed_create_edit_counter`,
    `mixed_create_edit_slug`, `source_free_project_one`,
    `source_free_project_two`, `source_free_project_three`, `revision_cli`,
    `revision_counter`, `revision_slug`, `preflight_cli`,
    `preflight_counter`, `preflight_release_feed`, `repair_cli`,
    `repair_counter`, `repair_release_feed`, `dependency_blocker_response`,
    `dependency_blocker_retry`, `dependency_blocker_cancel`,
    `blocker_answer_cli`, `blocker_answer_counter`,
    `blocker_answer_release`, `concurrent_source_cli`,
    `concurrent_source_counter`, and `concurrent_source_release`.
  - **Stage 5 handover:** begin only after the user grants explicit
    deterministic-test permission. Run and inspect the five deterministic
    commands above in order; remediate a failure through static failure-mode
    analysis before any rerun. After a clean deterministic gate, request a
    separate explicit real-LLM permission and run the two named nodes one at a
    time with trace review. Then request/confirm benchmark permission, preserve
    all historical attempts, execute the 30 cases sequentially under the new
    manifest/engine closure, bind reviewer-authored judgments, run the paired
    comparator, and obtain cutover authorization. Public route cutover, legacy
    manager/programmer pipeline removal, forbidden-symbol scans, documentation
    reconciliation, and independent code review remain the final Stage 5
    responsibilities. Stage 5 stays unchecked until all those records exist.
- Execution pause and progress correction, 2026-07-11:
  - The user halted subagent execution and all test/model activity after the
    live cohort repeatedly surfaced deterministic architecture gaps.
  - The implementation subagent was interrupted. Process inspection found no
    pytest, benchmark, or coding-agent LLM process; only pre-existing control
    console services remained.
  - Completed checkpoints Stage 1, Stage 2, Stage 3, and Stage 4 are preserved
    as immutable historical progress. Later findings belong to the new Stage
    4R remediation checkpoint and must not uncheck those stages.
  - Stage 5 is paused. At the pause boundary, six action-loop cases had
    reviewed canonical results under engine-contract digest
    `6ea6dc50b4711349b8299b63fd521913624eb41aad51acab7fdf7b41d391363a`;
    one additional case had completed execution and awaited review. Twenty-
    three locked cases had not yet produced a current canonical run.
  - All existing reviewed, provisional, retired, diagnostic, and raw workspace
    artifacts remain evidence. They are not permission to resume a benchmark.
  - The next active work is static architecture closure only: failure-mode
    matrix, identity/provenance audit, one-pass change design and implementation,
    repeated confidence assessment, and preparation of an unexecuted harness.
- Execution mode: explicit user-approved single-agent fallback requested on
  2026-07-11. This overrides the normal native-subagent execution requirement
  for this plan only.
- Phase B dependency evidence: completed plan at
  `development_plans/archive/completed/short_term/coding_agent_assessment_gap_phase_b_plan.md`.
- Phase C dependency evidence: completed plan at
  `development_plans/archive/completed/short_term/coding_agent_assessment_gap_phase_c_plan.md`;
  its completed evidence identifies three inspected `pipeline_v1` smoke cases
  only, so the Phase D v2 30-case baseline remains unrecorded.
- Approval and production-code authorization: user explicitly directed plan
  execution on 2026-07-11.
- Focused pre-implementation failures recorded: benchmark v2 selector,
  repository-index package, candidate overlay/recovery package, and
  action-loop parser package were absent before their corresponding tests.
- Focused implementation verification: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_benchmark_contracts.py
  tests\test_coding_agent_benchmark_contracts.py -q` passed 7/7 after the
  benchmark-only engine selector, `--engine-id`, and `--case-id` contract were
  added;
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_repository_index.py -q` passed 3/3;
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_candidate_recovery.py -q` passed 2/2;
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_action_loop_contracts.py -q` passed 2/2.
  Stages remain unchecked until their complete integration and evidence gates
  finish.
- Frozen benchmark manifest/version: `coding_agent_benchmark.v2`; SHA-256
  `a8214c567c8eafd4df1cb8c3a47eb4d51d30142d9a062407dee3ccbb88b171b3`
  for `tests/fixtures/coding_agent_benchmark/cases.jsonl` on 2026-07-11.
- `pipeline_v1` baseline artifact location: not recorded.
- Baseline progress: `pipeline_v1` case `source_backed_preflight_bugfix`
  completed on 2026-07-11 through the public `coding_run` entry point in
  101756 ms with two model calls. The durable result is
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_backed_preflight_bugfix.json`;
  the required human-readable review is
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_backed_preflight_bugfix.review.md`.
  The remaining 29 locked baseline cases are pending and Stage 1 remains
  unchecked.
- Baseline remediation evidence: `pipeline_v1` case
  `source_backed_cli_bugfix` timed out after 180012 ms and one model call. Its
  status is `blocked`, evaluator status is `not_applicable`, and its raw and
  human-readable evidence are under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`. This is a baseline
  failure, not a passed result; Phase D must preserve it in the comparison
  artifact and cannot waive the fixed timeout.
- Baseline progress: `pipeline_v1` case `source_backed_slug_bugfix` completed
  in 92692 ms with two model calls. Its result is
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_backed_slug_bugfix.json`
  and its human-readable review is
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_backed_slug_bugfix.review.md`.
- Baseline remediation evidence: `pipeline_v1` case `small_feature_cli` failed
  its expected `awaiting_approval` terminal state after 112726 ms and one model
  call. The raw result and review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`; this remains a failed
  locked baseline case and does not authorize evaluator changes.
- Baseline remediation evidence: `pipeline_v1` case `small_feature_counter`
  timed out after 180015 ms and three model calls. Its durable result and review
  are under `test_artifacts/coding_agent_benchmark/pipeline_v1/`; it remains a
  blocked baseline case with the timeout fixed at 180 seconds.
- Baseline progress: `pipeline_v1` case `small_feature_slug` reached its
  expected `awaiting_approval` state in 126722 ms with three model calls. Its
  result and human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`.
- Baseline remediation evidence: `pipeline_v1` case
  `mixed_create_edit_approval` timed out after 180007 ms and two model calls.
  Its raw result and review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/` as a blocked locked
  baseline case.
- Baseline progress: `pipeline_v1` case `mixed_create_edit_counter` completed
  in 161586 ms with two model calls. Its durable result and human-readable
  review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`.
- Baseline progress: `pipeline_v1` case `mixed_create_edit_slug` completed in
  124058 ms with two model calls. Its durable result and human-readable review
  are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`.
- Baseline remediation evidence: `pipeline_v1` case
  `source_free_project_one` timed out after 180003 ms and seven model calls.
  Its evaluator status is `not_applicable`; the raw result, durable trace, and
  human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`. This remains a blocked
  locked baseline case and keeps the fixed 180-second limit in force.
- Baseline execution correction: an initial `source_free_project_two` launch
  was stopped before a result after its root process and child process were
  misidentified as duplicate runners. Parent/child inspection established that
  this benchmark intentionally starts a child Python process. No artifact from
  the stopped launch was accepted; the current rerun is one root invocation
  with its expected child process.
- Benchmark-timeout amendment: user explicitly directed an increase from 180
  to 600 seconds on 2026-07-11. All `pipeline_v1` results that timed out at
  180 seconds are invalidated for comparison and must be rerun with the
  unchanged v2 manifest and 600-second limit. The active `repair_cli` process
  was stopped before a result and will be restarted under the amended limit.
- Baseline progress: amended `pipeline_v1` case `source_backed_cli_bugfix`
  reached the expected `completed` state in 163305 ms with two model calls.
  Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_backed_cli_bugfix/`;
  this supersedes its invalidated 180-second timeout.
- Baseline progress: amended `pipeline_v1` case `small_feature_counter`
  reached the expected `awaiting_approval` state in 199282 ms with two model
  calls. Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/small_feature_counter/`;
  this supersedes its invalidated 180-second timeout.
- Baseline progress: amended `pipeline_v1` case `mixed_create_edit_approval`
  reached the expected `completed` state in 145745 ms with two model calls.
  Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/mixed_create_edit_approval/`;
  this supersedes its invalidated 180-second timeout.
- Baseline remediation evidence: amended `pipeline_v1` case
  `source_free_project_one` reached the 600-second case limit after 13 model
  calls without its expected approval boundary. Its canonical result and
  partial durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_free_project_one/`.
  This is a non-passing amended baseline result; the remaining source-free
  cases must be inspected before any further timeout-contract change.
- Baseline remediation evidence: amended `pipeline_v1` case
  `source_free_project_two` reached the 600-second case limit after 18 model
  calls without its expected approval boundary. Its canonical result and
  partial durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_free_project_two/`.
  Two source-free cases now have this bounded non-passing result; the third
  must be inspected before any further timeout-contract change.
- Baseline remediation evidence: amended `pipeline_v1` case
  `source_free_project_three` reached the 600-second case limit after 19 model
  calls without its expected approval boundary. Its canonical result and
  partial durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/source_free_project_three/`.
  All source-free cases now exhibit the same bounded non-passing behavior;
  retain it in the frozen baseline rather than changing the contract again.
- Baseline progress: amended `pipeline_v1` case `preflight_release_feed`
  reached the expected `awaiting_approval` state in 245158 ms with two model
  calls. Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/preflight_release_feed/`;
  this supersedes its invalidated 180-second timeout.
- Baseline artifact reconciliation: all 30 `pipeline_v1` result records now
  have canonical `test_artifacts/coding_agent_benchmark/pipeline_v1/<case_id>/result.json`
  paths. The 11 non-timeout legacy results were byte-preserved into that layout;
  no completed or failed case was rerun. The v2 manifest lock is
  `test_artifacts/coding_agent_benchmark/manifest-lock.json` with the recorded
  SHA-256 and 600-second timeout. Stage 1 remains unchecked pending the final
  result-schema/comparator evidence required by its contract.
- Comparator precondition evidence: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_benchmark_contracts.py
  tests\test_coding_agent_benchmark_contracts.py -q` first failed because
  `compare_benchmark_results` was absent, then passed 8/8 after the offline
  comparator was added. It rejects a missing canonical paired engine result;
  no action-loop result set exists yet, so no comparison artifact or cutover
  judgment has been fabricated.
- Baseline remediation evidence: `pipeline_v1` case `repair_cli` was rerun
  under the 600-second limit and reached `blocked` rather than the expected
  `completed` state after 384360 ms and four model calls. Its canonical result
  is `test_artifacts/coding_agent_benchmark/pipeline_v1/repair_cli/result.json`;
  its durable trace and human-readable review are retained in the same
  engine-specific artifact tree. This meaningful failed result supersedes the
  stopped pre-amendment attempt and does not authorize evaluator changes.
- Baseline progress: `pipeline_v1` case `repair_counter` reached the expected
  `completed` state in 269580 ms with two model calls under the 600-second
  limit. Its canonical result is
  `test_artifacts/coding_agent_benchmark/pipeline_v1/repair_counter/result.json`;
  its durable trace and human-readable review are retained in the same
  engine-specific artifact tree.
- Baseline progress: `pipeline_v1` case `repair_release_feed` reached the
  expected `completed` state in 183185 ms with two model calls under the
  600-second limit. Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/repair_release_feed/`.
- Baseline progress: `pipeline_v1` case `dependency_blocker_response` reached
  the expected typed `blocked` state in 1351 ms with zero model calls. Its
  canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/dependency_blocker_response/`.
- Baseline progress: `pipeline_v1` case `dependency_blocker_retry` reached the
  expected typed `blocked` state in 1350 ms with zero model calls. Its
  canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/dependency_blocker_retry/`.
- Baseline progress: `pipeline_v1` case `dependency_blocker_cancel` reached
  the expected typed `blocked` state in 1340 ms with zero model calls. Its
  canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/dependency_blocker_cancel/`.
- Baseline remediation evidence: `pipeline_v1` case `blocker_answer_cli`
  reached `blocked` rather than the expected `completed` state in 228305 ms
  with two model calls. Its canonical result and durable trace are retained
  under `test_artifacts/coding_agent_benchmark/pipeline_v1/blocker_answer_cli/`;
  this meaningful failed result does not authorize evaluator changes.
- Baseline remediation evidence: `pipeline_v1` case `blocker_answer_counter`
  reached `blocked` rather than the expected `completed` state in 232583 ms
  with two model calls. Its canonical result and durable trace are retained
  under `test_artifacts/coding_agent_benchmark/pipeline_v1/blocker_answer_counter/`;
  this meaningful failed result does not authorize evaluator changes.
- Baseline remediation evidence: `pipeline_v1` case `blocker_answer_release`
  reached `blocked` rather than the expected `completed` state in 360724 ms
  with five model calls. Its canonical result and durable trace are retained
  under `test_artifacts/coding_agent_benchmark/pipeline_v1/blocker_answer_release/`;
  this meaningful failed result does not authorize evaluator changes.
- Baseline remediation evidence: `pipeline_v1` case `concurrent_source_cli`
  reached `failed` rather than the expected `awaiting_approval` state in
  184557 ms with one model call. Its canonical result and durable trace are
  retained under `test_artifacts/coding_agent_benchmark/pipeline_v1/concurrent_source_cli/`;
  this meaningful failed result does not authorize evaluator changes.
- Baseline progress: `pipeline_v1` case `concurrent_source_counter` reached
  the expected `awaiting_approval` state in 166757 ms with two model calls.
  Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/concurrent_source_counter/`.
- Baseline progress: `pipeline_v1` case `concurrent_source_release` reached
  the expected `awaiting_approval` state in 233466 ms with two model calls.
  Its canonical result and durable trace are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/concurrent_source_release/`.
- Baseline remediation evidence: `pipeline_v1` case
  `source_free_project_two` timed out after 180012 ms and seven model calls.
  Its evaluator status is `not_applicable`; the raw result, durable trace, and
  human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`. This remains a blocked
  locked baseline case and keeps the fixed 180-second limit in force.
- Baseline remediation evidence: `pipeline_v1` case
  `source_free_project_three` timed out after 180001 ms and twelve model
  calls. Its evaluator status is `not_applicable`; the raw result, durable
  trace, and human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`. This remains a blocked
  locked baseline case and keeps the fixed 180-second limit in force.
- Baseline remediation evidence: `pipeline_v1` case `revision_cli` reached a
  `failed` terminal state instead of the expected `awaiting_approval` state in
  85110 ms with one model call. Its raw result, durable trace, and
  human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`; this failed baseline
  result does not authorize evaluator or fixture changes.
- Baseline progress: `pipeline_v1` case `revision_counter` reached the expected
  `awaiting_approval` state in 166064 ms with two model calls. Its raw result,
  durable trace, and human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`.
- Baseline progress: `pipeline_v1` case `revision_slug` reached the expected
  `awaiting_approval` state in 141263 ms with two model calls. Its raw result,
  durable trace, and human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`.
- Baseline remediation evidence: `pipeline_v1` case `preflight_cli` reached a
  `failed` terminal state instead of the expected `awaiting_approval` state in
  67590 ms with one model call. Its raw result, durable trace, and
  human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`; this failed baseline
  result does not authorize evaluator or fixture changes.
- Baseline progress: `pipeline_v1` case `preflight_counter` reached the
  expected `awaiting_approval` state in 133535 ms with two model calls. Its raw
  result, durable trace, and human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`.
- Baseline remediation evidence: `pipeline_v1` case
  `preflight_release_feed` timed out after 180009 ms and one model call. Its
  evaluator status is `not_applicable`; the raw result, durable trace, and
  human-readable review are retained under
  `test_artifacts/coding_agent_benchmark/pipeline_v1/`. This remains a blocked
  locked baseline case and keeps the fixed 180-second limit in force.
- Repository-index implementation evidence: the Stage 2 focused suite initially
  failed because `_regex_rows` was absent. `regex_worker.py` and the bounded
  `regex` search mode were added; `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_repository_index.py
  tests\test_coding_agent_phase_d_candidate_recovery.py -q` then passed 6/6.
  The test artifacts use workspace-local SQLite snapshots and candidate state
  roots under pytest temporary workspaces.
- Delete/rename implementation evidence: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_patch_operations.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py -q` passed 7/7,
  including review-diff materialization for delete and rename.
- Action-loop implementation evidence: the same focused suite passed strict
  action validation, three-strike typed blocker behavior, prompt-safe semantic
  capabilities, unavailable effect handling, durable action/observation
  records, and typed missing-controller configuration.
- Controller deployment prerequisite: the local process has no
  `CODING_AGENT_ACTION_LOOP_LLM_BASE_URL`,
  `CODING_AGENT_ACTION_LOOP_LLM_API_KEY`, or
  `CODING_AGENT_ACTION_LOOP_LLM_MODEL` values. The new controller emits typed
  `controller_configuration_missing` evidence and does not break service
  import. Live action-loop and benchmark evaluation remain pending route
  configuration.
- Deterministic verification commands/results: not recorded.
- Stage 4 pre-implementation failure: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_coding_run_integration.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_coding_agent_phase_d_patch_operations.py -q` exited 1 because
  `tests/test_coding_agent_phase_d_coding_run_integration.py` is absent. The
  required public coding-run integration contract remains to be added.
- Architecture blocker (2026-07-11): Stage 4 requires a public coding-run
  integration test proving all objectives enter `action_loop_v1`, while the
  locked **Cutover Policy** and **Cutover Policy Enforcement** require public
  production dispatch to remain on `pipeline_v1` until the Stage 5 two-engine
  benchmark gate passes. Those conditions cannot both be true in one runtime
  without the forbidden engine switch/fallback. Additionally, this environment
  has no `CODING_AGENT_ACTION_LOOP_LLM_*` route values, so routing the public
  API to the loop now would deterministically block every new run. Execution
  stops here pending an explicit architecture decision: either move public
  dispatch/cutover verification after Stage 5 and keep Stage 4 evaluation-only,
  or authorize a different non-production integration boundary.
- Architecture decision resolution (2026-07-11): the executive decision adopts
  the existing private evaluation boundary for the complete Stage 4 lifecycle
  and defers public coding-run cutover until after the Stage 5 comparison gate.
  The blocker is resolved; Stage 1–3 are reopened because their current tests
  establish scaffolding rather than the plan's required contracts.
- Corrected Stage 1 manifest: the 30 JSONL rows now carry objective type,
  literal task text, fixture manifest digest, route-policy digest, acceptance
  checks, hard safety gates, and five rubric targets. The corrected manifest
  SHA-256 is `5817430601002ec87b7774c5d809dbb13b54599fdc8762555f9a92d7da3ff623`.
  This supersedes the prior lock and invalidates all prior `pipeline_v1`
  artifacts for paired comparison; rerun is required after the immutable
  result/judgment schema is completed.
- Corrected benchmark contract verification: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_benchmark_contracts.py
  tests\test_coding_agent_benchmark_contracts.py -q` passed 9/9 after the
  manifest and immutable result metadata were upgraded. The historical results
  are retained but must not be used by the corrected comparator.
- Corrected baseline trace inventory: all 30 selected `pipeline_v1` cases have
  a canonical provisional result at
  `test_artifacts/coding_agent_benchmark/pipeline_v1/<case_id>/result.json`
  with its `run.json` and `events.jsonl` trace paths. The 2026-07-11 first
  pass has 21 passing terminal evaluations, seven terminal-state mismatches
  (`small_feature_cli`, `revision_cli`, `preflight_cli`, `repair_cli`, and all
  three `blocker_answer_*` cases), and two bounded timeouts
  (`source_free_project_one` and `source_free_project_two`). Every passing
  provisional result records the judgment `Reached the locked terminal state;
  inspect retained trace evidence.`; mismatches record `Terminal state differs
  from the locked acceptance target.`; timeouts record `Timed out before
  reaching the locked terminal state.` These artifacts are evidence for the
  forthcoming trace-derived hard-gate and rubric finalization, not Stage 1
  sign-off.
- Harness correction: `repair_cli` initially produced a 19 ms rejected result
  because `verify_repair` start requests lacked the required structured
  approval and execution specs. That artifact is preserved at
  `test_artifacts/coding_agent_benchmark/pipeline_v1/repair_cli_harness_defect_rejected/`.
  The harness now supplies those initial fields; its focused benchmark suites
  passed 10/10, and the one necessary rerun produced the 325775 ms typed
  `blocked` canonical result recorded above.
- Comparator remediation: the focused comparator regression first failed
  because a false hard gate was accepted. The comparator now validates the
  locked manifest, objective, fixture digest, route-policy digest, acceptance
  keys, hard-gate keys/outcomes, and rubric shape before scoring; it reports
  exact rubric averages by category. `venv\Scripts\python -m pytest
  tests\test_coding_agent_benchmark_contracts.py
  tests\test_coding_agent_phase_d_benchmark_contracts.py -q` then passed
  11/11. New case results derive source-immutability and approval-boundary
  outcomes from durable source/run evidence and score non-target results below
  2 with trace-citing judgment notes. Existing provisional results await the
  same offline trace finalization without a second model invocation.
- Stage 1 sign-off: all 30 canonical results now validate under the upgraded
  schema, each retains `result.provisional.json` and `context_manifest.json`,
  every recorded hard safety gate passes, and the agent-authored review is
  `test_artifacts/coding_agent_benchmark/pipeline_v1/baseline_trace_review.md`.
  The complete deterministic verification command passed 11/11; the locked
  baseline outcome is 21 passed, seven failed terminal evaluations, and two
  bounded timeout results. Stage 1 is complete; the parent must reread this
  full plan before Stage 2 work.
- Stage 2 targeted failure and remediation: the strengthened focused suite
  initially failed because published snapshot databases lacked the required
  `chunk_fts` FTS5 projection. `repository_index.builder` now creates and
  populates that projection and `search_snapshot` routes literal search through
  a phrase-safe FTS5 query. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_repository_index.py
  tests\test_coding_agent_phase_d_candidate_recovery.py -q` passed 8/8,
  including the retained-prior-snapshot and mismatched-write recovery tests.
  Stage 2 remains open pending atomic publication and base/overlay merge
  evidence.
- Stage 2 overlay/recovery remediation: a corrupted `candidate_written`
  operation initially raised without restoring its pre-write candidate state,
  and the overlay lacked a merged base-snapshot view. Recovery now journals the
  prior content and atomically restores/removes the candidate file before
  persisting `rolled_back`; `CandidateOverlay.merge_base_rows` suppresses every
  overlay-owned stale base path and returns live overlay rows in deterministic
  order. The focused Stage 2 command now passes 9/9. Atomic publication and
  storage reclamation/pinning evidence remain required before Stage 2 sign-off.
- Stage 2 sign-off: the missing `repository_index.storage` boundary was added
  after its focused reclamation test failed. Builder publication now writes a
  `building` SQLite file and replaces the final snapshot only after its
  complete transaction; storage reclaims only complete snapshots not pinned by
  a non-archived ledger or active cursor. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_repository_index.py
  tests\test_coding_agent_phase_d_candidate_recovery.py -q` passed 10/10.
  Stage 2 is complete; reread this full plan before Stage 3 work.
- Stage 2 reopened-schema remediation: the prior store exposed only text rows
  and literal/regex search. The published snapshot now contains canonical
  `file`, `chunk`, `chunk_fts`, `symbol`, and `import_edge` rows; Python
  symbols/imports are indexed; literal, regex, symbol, and path search return
  prompt-safe canonical results; and cursors bind snapshot, normalized query,
  path glob, and candidate overlay revision. The focused repository-index and
  candidate-recovery command passed 11/11 after the new test first failed on
  the missing `overlay_revision` parameter. Stage 2 remains open for streamed
  resource-policy and complete candidate-overlay lifecycle verification.
- Stage 2 resource-policy remediation: the index had no explicit oversized
  safe-file result. Shared context-budget constants now define the Phase D
  index/search/read resource vocabulary, and snapshot construction checks file
  size before decoding content. An oversized eligible file returns the typed
  `index_resource_exhausted` blocker with `MAX_INDEX_FILE_BYTES`; it is not
  silently excluded. The focused repository-index and candidate-recovery suite
  passed 12/12 after the new test first failed on the absent bound.
- Stage 2 candidate-rename remediation: candidate state had no durable rename
  operation, so a rename could not be resumed or revision-bound. The state
  journal now records prepared/candidate-written/committed rename transitions,
  atomically moves the candidate path, and verifies persisted content before
  advancing revision during recovery. The new recovery test first failed on
  the missing `prepare_rename` method; `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_candidate_recovery.py -q` passed 5/5.
  Stage 2 remains open for the remaining full overlay lifecycle contracts.
- Stage 3 targeted failure and remediation: the new canonical-record test
  initially failed because the patch boundary had no operation record or
  digest API. `CanonicalPatchOperationRecord` now owns delete/rename source,
  target, source-hash, candidate-revision, result-hash, and content-hash
  fields; `build_canonical_operation_records` validates source/target state and
  `canonical_proposal_digest` hashes their ordered canonical JSON. The Stage 3
  focused command passed 8/8. Durable generic action-loop lifecycle and
  review/apply propagation remain required before Stage 3 sign-off.
- Stage 3 lifecycle remediation: the durable-loop contract initially failed
  because `run_recorded_actions` only wrote JSONL actions/observations. It now
  writes bounded `state.json`, `working_notes.json`, and
  `context_manifest.json` alongside those ordered artifacts while retaining
  semantic controller ownership. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_coding_agent_phase_d_patch_operations.py -q` passed 9/9.
  Canonical operation propagation through review and approved apply remains
  required before Stage 3 sign-off.
- Stage 3 binding remediation: the approved-apply provenance test initially
  failed because no canonical digest/revision validator existed. The canonical
  operation owner now rejects a digest that differs from the ordered records
  and any record whose expected candidate revision differs from the current
  candidate. The Stage 3 focused command passed 10/10. The apply request and
  response still need to carry and enforce those fields before sign-off.
- Stage 3 apply enforcement: `materialize_managed_candidate` now validates
  optional evaluation-time canonical operation records, proposal digest, and
  candidate revision before authorization or filesystem work; a missing,
  malformed, stale, or digest-mismatched binding returns a typed rejected
  result. The Stage 3 focused command passed 11/11. Successful apply evidence
  still needs to echo the validated canonical records and digest before sign-off.
- Stage 3 successful-apply evidence: managed apply responses now echo the
  canonical records and digest after a validated canonical request succeeds.
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_patch_operations.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_coding_agent_phase5_patch_apply_contracts.py -q` passed 19/19,
  confirming the provenance addition retained approval and source-safety
  behavior. Stage 3 remains open pending an end-to-end canonical request
  materialization proof.
- Stage 3 sign-off: a real managed-copy delete request carrying its canonical
  record and digest succeeded and returned the same provenance in the apply
  result. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_patch_operations.py -q` passed 5/5; the
  combined Stage 3 and existing managed-apply contracts passed 19/19. Stage 3
  is complete; reread this full plan before Stage 4 work.
- Stage 3 reopened-protocol remediation: a new strict-parser contract first
  accepted a semantic `run` action carrying a command field, an incomplete
  search action, and an oversized reason. `coding_action.v1` now rejects
  unknown action-specific keys, validates the closed per-action argument
  vocabulary, rejects command-bearing run requests, and bounds reason and
  working-note text before dispatch. The lifecycle fixture was corrected to
  use a valid finish payload. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_action_loop_contracts.py -q` passed 8/8.
  Stage 3 remains open for controller-loop resume, candidate mutation, and
  semantic execution-policy contracts.
- Stage 3 read-dispatch remediation: `read` had been advertised as a semantic
  capability but returned `action_unavailable`. The immutable snapshot now
  provides bounded one-based line reads with complete-snapshot validation, and
  the action executor projects prompt-safe read evidence without host paths.
  The new contract first failed with `outcome=unavailable`; the combined
  action-loop and repository-index verification passed 16/16 after the
  executor was added. Stage 3 remains open.
- Stage 4 targeted failure and remediation: the required private integration
  test initially failed because `evaluation.py` had no continuation entry
  point. `continue_evaluation_coding_run` now keeps the locked engine selector
  private, delegates only `pipeline_v1` continuation to the existing durable
  lifecycle, and returns typed isolation/configuration results for other
  engines. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_coding_run_integration.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_coding_agent_phase_d_patch_operations.py -q` passed 13/13.
  The final private action-loop start/resume lifecycle remains required.
- Stage 4 private-start remediation: the new isolated-start test initially
  reached the action-loop placeholder failure. `run_evaluation_coding_run` now
  creates an evaluation-root run id, exposes objective-scoped semantic
  capabilities, invokes the single controller, persists the existing loop
  artifacts, and projects only finish/block outcomes without changing public
  dispatch. The Stage 4 focused command passed 14/14. Private action-loop
  approval and blocker-resume projection remain required before sign-off.
- Stage 4 sign-off: private continuation now writes structurally valid approval
  evidence only under the evaluation run root and projects persisted loop
  status without exposing an engine selector. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_coding_run_integration.py -q` passed 3/3.
  Stage 4 is complete; reread this full plan before Stage 5 work.
- Deterministic Phase D gate: `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_repository_index.py
  tests\test_coding_agent_phase_d_candidate_recovery.py
  tests\test_coding_agent_phase_d_patch_operations.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_coding_agent_phase_d_coding_run_integration.py
  tests\test_coding_agent_phase_d_benchmark_contracts.py -q` passed 30/30.
- Stage 5 route-configuration blocker: the current process has no configured
  `CODING_AGENT_ACTION_LOOP_LLM_BASE_URL`,
  `CODING_AGENT_ACTION_LOOP_LLM_API_KEY`, or
  `CODING_AGENT_ACTION_LOOP_LLM_MODEL`. A real controller call therefore
  returns `controller_configuration_missing`, so the required one-at-a-time
  real-LLM cases, `action_loop_v1` 30-case benchmark, comparator artifact,
  and post-gate public cutover cannot be truthfully run. Public dispatch remains
  on `pipeline_v1` as required. Resume Stage 5 only after an operator supplies
  all three route values to the process.
- Stage 5 configuration-contract remediation: the action-loop route now has
  optional `config.py` base URL, API-key, model, completion-budget, and
  thinking settings; the controller reads those settings only at invocation;
  and startup diagnostics list the route as non-required before cutover.
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_llm_interface_route_report.py tests\test_config.py -q` passed
  76/76. The focused recheck of the route-report and action-loop suites passed
  12/12, and `git diff --check` passed. The actual route values remain absent,
  so no live invocation or benchmark artifact was fabricated.
- Stage 5 route availability update: on 2026-07-11 the operator copied the
  configured programmer route into the private action-loop route in `.env`.
  A fresh process verified all three route settings match without logging the
  values. Real-controller cases may now begin one at a time; this update does
  not change public `pipeline_v1` dispatch or satisfy the benchmark gate.
- Reopened Stage 3/4 remediation: the prior evaluation engine performed one
  controller turn without source preparation, a pinned snapshot, candidate
  mutation, or durable private continuation. The engine now resolves the
  source, creates a candidate, pins the base snapshot, persists raw/parsed
  action and context evidence, validates revision/hash-bound edits, materializes
  canonical review provenance, preserves verbatim blocker responses, and
  resumes approved verification failures through the same private loop. The
  semantic `run` action selects only pre-existing approved structured checks
  and managed apply references; it accepts no command text or selector
  synthesis. `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_candidate_recovery.py
  tests\test_coding_agent_phase_d_patch_operations.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py
  tests\test_coding_agent_phase_d_coding_run_integration.py
  tests\test_coding_agent_phase_d_benchmark_contracts.py
  tests\test_coding_agent_benchmark_contracts.py -q` passed 42/42.
- Stage 2 current sign-off verification: after freezing the evaluation engine
  runtime closure at SHA-256
  `6ea6dc50b4711349b8299b63fd521913624eb41aad51acab7fdf7b41d391363a`,
  `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_repository_index.py
  tests\test_coding_agent_phase_d_candidate_recovery.py -q` passed 30/30.
  The named gate covers canonical snapshot rows and publication, all search
  modes and cursor identities, streamed large-file and secret exclusion,
  typed resource exhaustion, pin/release/reclamation, candidate overlay
  search, and durable rollback/recovery for edit, delete, and rename. The
  superseded reopened checklist note is replaced by this evidence.
- Stage 3 current sign-off verification: with the same frozen evaluation
  engine closure, `venv\Scripts\python -m pytest
  tests\test_coding_agent_phase_d_patch_operations.py
  tests\test_coding_agent_phase_d_action_loop_contracts.py -q` passed 37/37.
  The named gate includes canonical digest/revision/apply provenance,
  source-free create-to-edit/delete/rename compilation, strict action-specific
  parsing, current-candidate read/search, trusted semantic run selection,
  durable orphan reconciliation, exact run and wall-time budgets, and
  persistence before the next controller turn. The superseded reopened
  checklist note is replaced by this evidence.
- Live controller evidence: the individually run
  `test_controller_live_chooses_targeted_repository_search` initially rejected
  numeric `action_id=0`; the parser remained strict and the static prompt was
  clarified. The inspected passing trace is
  `test_artifacts/llm_traces/coding_agent_phase_d_controller_targeted_search_20260711T065551Z.json`;
  the agent-authored comparison review is
  `test_artifacts/llm_traces/coding_agent_phase_d_controller_targeted_search_review.md`.
- Live graph evidence: the one-at-a-time `action_loop_v1` benchmark case
  `small_feature_slug` reached its locked `awaiting_approval` terminal state
  in 22075 ms with five controller calls. Its result and agent-authored review
  are under
  `test_artifacts/coding_agent_benchmark/action_loop_v1/small_feature_slug/`.
  The review records that the locked fixture does not prove full
  semantic-preservation coverage; it does not waive that limitation or the
  remaining benchmark gate.
- Patched handoff verification: included in the 42/42 reopened Stage 3/4
  deterministic gate above; it covers isolated start, source/index handoff,
  candidate review, approval/apply, blocker response, verification-failure
  resume, and approved semantic run selection.
- Real-LLM per-case trace paths and judgments: not recorded.
- Benchmark comparison artifact and threshold judgment: not recorded.
- Independent plan reviewer and findings: `/root/independent_plan_review`,
  2026-07-11. Remediated draft blockers: benchmark v2/comparator, valid
  snapshot schema, candidate journal recovery, post-cutover public boundary,
  delete/rename artifact contract, source-free candidate identity, closed
  blocker mapping, regex/resource policy, index privacy policy, granular
  test-first stages, cutover enforcement matrix, and dependency evidence. The
  re-review additionally required and this revision records the final-owner
  dispositions for legacy imports, deterministic execution planning with one
  post-source-resolution controller, controller route ownership, and complete
  streaming secret scanning.
- Independent code reviewer and findings: not recorded; required after planned
  implementation verification and before any completion sign-off.
- Cutover/drain record: not recorded.
- Removed production paths/caps/tests: not recorded.
- Post-cutover scan commands/results: not recorded.
- Documentation reconciliation: not recorded.
- Final sign-off and archive path: not recorded.

## Risks

| Risk | Mitigation |
|---|---|
| Index serves stale source content | Content-manifest snapshot identity, immutable complete snapshots, pinned run identity, and candidate revision evidence. |
| Index construction is expensive for large repositories | Persistent local reuse, streamed chunking, content-hash reuse, resumable incomplete builds, and lazy per-source creation. |
| Excluded or secret content enters the index | Versioned deterministic exclusion policy before hashing/chunking, safe-text classification, source-scope validation, and privacy hard gates. |
| Controller loops without useful progress | Exact turn/wall/run budgets, repeated-signature detection, durable observations, and typed blocker termination. |
| Local LLM loses task state as observations grow | Deterministic 50k context reducer, bounded working notes, changed-path summary, failure priority, and artifact-backed old evidence. |
| Malformed JSON stalls the run | One-action parser, bounded invalid observations, and three-strike typed blocker. |
| Delete removes the wrong file | Normalized source scope, exact hash/revision preconditions, protected-path checks, candidate-only mutation, and review/approval binding. |
| Rename overwrites or escapes scope | Absent-target, same-source, collision, safe-file, hash/revision, and lock validation. |
| Base index disagrees with edited candidate | Transactional overlay updates, tombstones, pinned revisions, and merged search contract. |
| Model smuggles commands through run | Semantic closed schema and deterministic Phase B planning with no command field. |
| Old pipeline remains as hidden fallback | One cutover, forbidden-symbol/import scans, no runtime flags, and independent review. |
| Existing tests bias implementation toward removed topology | Define verification from target contracts, inventory tests only after new boundaries exist, and remove topology-owned cases at cutover. |
| Benchmark variance masks regression | Frozen source/task/model inputs, one-at-a-time traces, hard safety gates, explicit behavioral rubric, and retained comparison artifacts. |
