# coding_agent_assessment_gap_phase_d_plan

## Summary

- Goal: Replace the coding agent's fixed repository-exploration and
  proposal/repair pipelines with one bounded, persistent JSON action loop that
  can read, search, edit, run approved verification, delete files, rename
  files, and use a complete persistent repository index without file, wave, or
  report-count exploration caps.
- Plan class: high-risk migration
- Status: draft
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`
- Dependencies: Complete Phase B and Phase C first. Phase D consumes Phase B's
  proposal-bound candidate workspace, source-free preflight policy, execution
  planner, repair failure bundle, and revision binding. It consumes Phase C's
  durable continuation context, typed blocker resume targets, operation locks,
  and versioned 30-case benchmark seam.
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

- Phase D remains `draft` until Phase B and Phase C are complete.
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
- Index all safe text files in source scope. There is no maximum indexed-file
  count and no early stop after a fixed number of files.
- Stream large text files into bounded line-aware chunks so index construction
  does not require loading an entire repository or file into memory.
- Initial symbol extraction supports Python through `ast`. All safe text still
  receives full-text and path indexing when no language-specific extractor is
  available.
- Keep the index deterministic and local. The model never receives the
  database path, raw SQLite rows, or unrestricted dump of repository content.

### 5. Index Schema And Search Contract

The SQLite database owns these canonical entities:

```text
snapshot(snapshot_id, source_identity_hash, manifest_digest,
         schema_version, exclusion_policy_version, status, created_at)
file(file_id, repo_path, content_sha256, byte_size, language, line_count)
chunk(chunk_id, file_id, start_line, end_line, content)
chunk_fts(content, repo_path, content='chunk', content_rowid='chunk_id')
symbol(symbol_id, file_id, qualified_name, symbol_kind,
       start_line, end_line, signature)
import_edge(edge_id, file_id, imported_name, target_repo_path)
```

- `search` supports `literal`, `regex`, `symbol`, and `path` modes through one
  deterministic interface.
- FTS handles ranked literal search. Regex executes over indexed safe-text
  chunks with a deterministic timeout. Symbol and path modes use their
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

### 6. Candidate Overlay

- Keep the immutable source snapshot unchanged during a coding run.
- Maintain a run-local overlay index for every created, modified, deleted, or
  renamed path in the Phase B candidate.
- A successful edit updates the overlay transactionally with the candidate
  revision.
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
  history. Before production cutover:
  - allow already-running work to reach a stable terminal or blocked state;
  - reject new continuation mutations during the cutover lock window;
  - preserve old artifacts as read-only audit evidence;
  - resume eligible blocked work by creating an explicit `action_loop_v1`
    continuation event bound to the latest validated proposal/candidate state.
- Eligible continuation requires a valid source identity, proposal digest,
  candidate revision, and current Phase B/Phase C authorization state.
- Runs lacking those bindings remain blocked with a typed explanation and a
  fresh-run action. Deterministic code must not infer missing provenance.
- Build repository indexes lazily on first post-cutover use and reuse them
  afterward. Do not eagerly scan unrelated repositories during deployment.
- Stamp all index, loop, action, observation, and patch-operation artifacts with
  explicit schema versions.

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
  - `README.md`
- New deterministic and real-LLM target-state verification owned by these
  packages and by the public coding-run route.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`
  - route all three objectives through the action-loop engine;
  - remove capped proposal fallback evidence assembly.
- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`
- `src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py`
- `src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py`
  - persist and resume action-loop state through the existing run API.
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
- Coding-agent architecture, subsystem READMEs, and ICD/reference documents
  that describe generation, repair, repository reading, patch operations, or
  execution.

### Remove At Cutover

- Model orchestration, prompts, and topology-only models under:
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/`;
  - `src/kazusa_ai_chatbot/coding_agent/code_writing/`;
  - `src/kazusa_ai_chatbot/coding_agent/code_modifying/`.
- Move genuinely shared deterministic validation to its final owner before
  removing these packages.
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
  external search service. SQLite FTS5 plus deterministic symbol/path/regex
  search is the Phase D index.
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

1. Confirm Phase B and Phase C completion evidence and approve Phase D.
2. Freeze and version the Phase C 30-case benchmark manifest and `pipeline_v1`
   baseline artifacts before changing production dispatch.
3. Add repository-index identity, exclusion policy, storage, full build,
   incremental reuse, search, and atomic completion behavior.
4. Add the candidate overlay and prove create/replace/delete/rename visibility
   against one pinned base snapshot.
5. Add canonical delete/rename patch operations, candidate mutation, review
   rendering, digest binding, and final apply validation.
6. Add the versioned action and observation models, strict parser, durable loop
   state, recovery rules, context reducer, and budget enforcement.
7. Add read/search/note/finish/block handlers and complete the read-only
   objective in the evaluation-only engine.
8. Add edit and semantic run handlers, candidate finalization, Phase B
   preflight, and same-loop repair observations.
9. Add coding-run resume and Phase C typed-blocker integration.
10. Create target-state deterministic, handoff, and real-LLM verification from
    this plan's contracts. Inventory existing tests only now, then remove or
    replace topology-bound cases without carrying old architecture forward.
11. Run `action_loop_v1` through the benchmark one case at a time and inspect
    every durable trace.
12. Complete independent review and remediate all critical/high findings.
13. Acquire the cutover locks, drain or stabilize existing runs, route all
    objectives to the new engine, and remove superseded production paths in the
    same change.
14. Run post-cutover target-state verification, documentation reconciliation,
    forbidden-symbol scans, and lifecycle closeout.

## Execution Model

- Parent agent owns plan state, architecture decisions, shared-contract
  integration, benchmark judgment, production cutover, and final sign-off.
- Use parallel implementation agents only for disjoint worktrees/surfaces:
  repository index, patch delete/rename, action-loop core, and verification
  artifacts.
- A child agent must not edit shared public models, production dispatcher, plan
  status, registry, or cutover code without a parent-assigned file boundary.
- Parent integrates shared models and resolves cross-surface contract changes.
- Benchmark cases run sequentially because each real-LLM trace requires
  individual inspection.
- If native subagents are unavailable, stop before execution and request the
  user's explicit fallback authorization.

## Progress Checklist

- [ ] Phase B and Phase C completion evidence confirmed.
- [ ] Phase D approved and production-code authorization recorded.
- [ ] Benchmark manifest/version and `pipeline_v1` baseline frozen.
- [ ] Persistent repository index implemented and documented.
- [ ] Complete-source indexing and incremental reuse verified.
- [ ] Candidate overlay implemented and transactionally revision-bound.
- [ ] Delete and rename operations implemented across candidate, patch, review,
  digest, approval, verification, and apply boundaries.
- [ ] Strict action/observation schemas and parser implemented.
- [ ] Durable action-loop state and crash recovery implemented.
- [ ] Context reduction and resource/no-progress budgets implemented.
- [ ] Read-only objective operational in the evaluation engine.
- [ ] Proposal and repair objectives operational in the evaluation engine.
- [ ] Phase B execution/preflight and failure bundles integrated as actions and
  observations.
- [ ] Phase C blocker/resume/lock contracts integrated.
- [ ] Target-state deterministic and handoff verification complete.
- [ ] Real-LLM node/loop cases run and inspected one at a time.
- [ ] Phase C benchmark gate passed.
- [ ] Independent review complete with critical/high findings remediated.
- [ ] Production big-bang cutover complete.
- [ ] Superseded model pipelines, caps, tests, and documentation removed.
- [ ] Post-cutover verification and forbidden-symbol scans complete.
- [ ] Execution Evidence complete and plan archived.

## Verification

### Deterministic Contract Verification

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

- Search production source for imports and references to removed model
  pipelines and fixed cap symbols.
- Search prompts and public models for old action aliases and native-tool-call
  assumptions.
- Confirm the production dispatcher has exactly one route per objective and no
  `pipeline_v1` environment switch or fallback.
- Confirm docs and README diagrams match the final production boundary.

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

- Status: not started.
- Phase B dependency evidence: not recorded.
- Phase C dependency evidence: not recorded.
- Approval and production-code authorization: not recorded.
- Frozen benchmark manifest/version: not recorded.
- `pipeline_v1` baseline artifact location: not recorded.
- Repository-index implementation evidence: not recorded.
- Delete/rename implementation evidence: not recorded.
- Action-loop implementation evidence: not recorded.
- Deterministic verification commands/results: not recorded.
- Patched handoff verification commands/results: not recorded.
- Real-LLM per-case trace paths and judgments: not recorded.
- Benchmark comparison artifact and threshold judgment: not recorded.
- Independent reviewer and findings: not recorded.
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

