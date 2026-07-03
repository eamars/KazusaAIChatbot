# coding_agent phase 0 fetching plan

## Summary

- Goal: Create the standalone `code_fetching` subagent package and ICD so
  upstream coding-agent stages can resolve source locations, managed storage,
  and repository metadata without reading code for answers.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: additive standalone module only; no Kazusa service,
  L2d, background-work, result-ready, or code-reading integration in Phase 0.
- Highest-risk areas: fetched-code storage, active-checkout mutation, unsafe
  path exposure, ambiguous file-vs-repo source intent, private/auth repo
  refusal, and unclear subagent ICD boundaries.
- Acceptance criteria: direct tests can call the `code_fetching` public
  subagent entrypoint, resolve repo/file/tree/raw GitHub and local-checkout
  inputs, reject or defer unsupported source classes deterministically, store
  managed public clones under the configured workspace, return a safe
  `CodeFetchingResult`, and pass a 10-source public internet smoke run.

## Context

The coding-agent architecture is being split into smaller implementation
stages. Phase 0 owns only source resolution and managed code storage. Phase 1
will build code reading on top of Phase 0 output.

The user clarified these requirements:

```text
Each subagent shall have the corresponding README.md as ICD to interface with
upstream agent.
```

The user also clarified that `code_fetching` must support source-scope routing
because the input may point to a whole repository, a directory/tree, or a
single file.

```text
where the code will be stored after fetching
```

The user later raised the Phase 0 bar:

```text
brainstorm all possible inputs to the code fetching agent
generate a list of test criteria
propose a configurable workspace to store the fetched code
learn how a generic coding agent fetches code
```

The reviewed source model is Codex-like: coding work is bound to an explicit
repository/workspace, runs inside bounded filesystem and network rules, stores
managed checkouts under a dedicated agent root, and reports traceable evidence
rather than treating downloads as free-form browsing.

Phase 0 must implement this direct path:

```text
CodeFetchingRequest
  -> code_fetching.run
  -> deterministic source extraction and validation
  -> deterministic source-scope routing
  -> local checkout resolution or managed clone
  -> CodeFetchingResult
```

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing source-scope routing prompts,
  LLM call counts, context budgets, or subagent boundaries.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python tests or source files that contain
  CJK text.

## Mandatory Rules

- Implementation is approved for this plan only by the user's 2026-06-20
  instruction to implement without subagents. Keep `Status` as `in_progress`
  during execution.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use single-agent fallback execution for this plan only. The user explicitly
  requested implementation without subagents, so the parent agent owns tests,
  production code, verification, review, evidence, and lifecycle updates.
- Use `venv\Scripts\python.exe` for Python commands.
- Never read `.env` during implementation or verification.
- Keep Phase 0 independent from Kazusa runtime integration. Do not modify L2d,
  action-spec, background-work, result-ready cognition, service delivery,
  adapters, persistence, consolidation, or scheduler code.
- Keep `code_fetching` as a subagent package with its own
  `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md` ICD.
- Keep source-scope routing deterministic in Phase 0. Deterministic code owns
  URL parsing, source-kind classification, command construction, path safety,
  storage layout, and refusal.
- Do not mutate the active project checkout. Existing local checkouts may be
  read for metadata only.
- Do not parse `workspace_root` from user text. It must come from the typed
  request or test/config harness.
- Do not expose credentials, raw command output, `.git` internals, or
  unmanaged absolute paths in public result fields.

## Must Do

- Create `src/kazusa_ai_chatbot/coding_agent/` as the standalone package root
  if it does not already exist.
- Add `src/kazusa_ai_chatbot/coding_agent/README.md` with staged module
  ownership and a Phase 0 note that only `code_fetching` is implemented.
- Add `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md` as the
  upstream ICD for the fetching subagent.
- Add `code_fetching.run(request: CodeFetchingRequest) -> CodeFetchingResult`
  as the subagent public entrypoint.
- Implement deterministic extraction and parsing for public GitHub repository,
  `.git`, tree, blob, and raw-file URLs found in explicit fields or question
  text.
- Implement deterministic `owner/repo` parsing for `repo_hint`.
- Implement local checkout and local file/directory path handling through
  explicit `local_root_hint` and `local_path_hint` fields.
- Implement a bounded source-scope router that classifies supported inputs as
  `repository`, `directory`, or `file`, and returns `needs_user_input` for
  ambiguous supported candidates.
- Implement local-checkout matching and managed public-clone storage.
- Implement metadata checks that prevent overwriting a managed workspace path
  when `metadata.json` does not match the requested source.
- Add deterministic tests for URL parsing, source-scope decisions, unsupported
  inputs, local checkout matching, managed clone path construction, metadata
  mismatch refusal, storage safety, command argument safety, and response
  shape.
- Add a live-internet smoke test or diagnostic that resolves 10 public GitHub
  code sources through the public `code_fetching.run(...)` interface.

## Deferred

- Do not implement `code_reading`.
- Do not implement `answer_code_question`.
- Do not add the top-level coding supervisor loop.
- Do not connect `coding_agent` to background-work, L2d, action-spec, service,
  result-ready cognition, adapters, persistence, consolidation, or scheduler.
- Do not remove or decommission `text_artifact`.
- Do not implement code writing, patch proposal, patch apply, execution, Docker
  execution, package installation, arbitrary shell, private GitHub, SSH clone,
  credential handling, authenticated repositories, or `web_agent3` calls.
- Do not implement GitLab, Bitbucket, Azure DevOps, package registry, release
  archive, Gist, paste, generic raw URL, issue, pull request, discussion, or
  uploaded-archive fetching in Phase 0. Return a tested unsupported result for
  these input classes.
- Do not add a Phase 0 LLM source router. Add LLM-based disambiguation only
  after deterministic supported inputs and Phase 1 reading expose a concrete
  failure that requires semantic routing.

## Cutover Policy

Overall strategy: additive standalone subagent. Phase 0 has no Kazusa runtime
cutover.

| Area | Policy | Instruction |
|---|---|---|
| `coding_agent` package root | additive | Create only the root scaffolding needed by `code_fetching`. |
| `code_fetching` subagent | additive | Implement a package with README ICD and direct tests. |
| Kazusa runtime paths | no-op | Do not wire fetching into service, L2d, background work, or result-ready delivery. |
| Existing background workers | no-op | Keep existing production behavior unchanged. |
| Repository mutation | bigbang no-op | Phase 0 forbids mutation of existing local checkouts. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not expand Phase 0 into reading or service integration.
- For no-op areas, leave existing runtime behavior unchanged.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

After Phase 0, upstream coding-agent code can call `code_fetching.run(...)`
directly and receive one resolved source:

```text
CodeFetchingRequest
  -> CodeFetchingResult(repository, source_scope, storage, trace_summary)
```

The `code_fetching` README is the ICD. It defines what upstream agents may
pass in, what they receive, and what storage guarantees are valid.

Managed public clones are stored outside the active Kazusa checkout by default:

```text
<workspace_root>/
  repos/
    github/
      <owner>/
        <repo>/
          refs/
            <ref_key>/
              checkout/
              metadata.json
  locks/
    <cache_key>.lock
  tmp/
    <cache_key>.<pid>/
```

`workspace_root` comes from `CodeFetchingRequest.workspace_root` in tests and
future worker configuration. Standalone direct use may fall back to an OS temp
directory named `kazusa_coding_agent`, but production/background-worker
integration must pass an explicit configured workspace root. Tests must pass
`tmp_path`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stage split | Phase 0 implements fetching only. | Fetching owns storage and source scope; reading depends on this contract. |
| Subagent packaging | `code_fetching` is a folder with README ICD. | It is an owned subagent boundary, not a helper file. |
| Storage | Use existing local checkouts read-only or managed public clones. | Keeps active worktrees safe and gives reading a stable root. |
| Workspace default | Use request/config workspace roots; allow temp-dir fallback only for standalone direct use. | Avoids polluting the Kazusa repo while keeping future worker storage explicit. |
| Source routing | Use deterministic URL/path shape routing in Phase 0. | Repo/tree/blob/raw/local-path scope is knowable without an LLM; ambiguity should ask for clarification. |
| URL parsing | Deterministic code parses GitHub URL shape. | Safety and storage paths must not depend on LLM output. |
| Unsupported inputs | Return `rejected` or `needs_user_input` instead of guessing. | Robust fetching means unsupported sources are explicit contract outcomes. |

## Supported And Unsupported Input Matrix

| Input class | Example | Phase 0 result |
|---|---|---|
| GitHub repository URL | `https://github.com/eamars/KazusaAIChatbot` | Support as `repository`. |
| GitHub clone URL | `https://github.com/eamars/KazusaAIChatbot.git` | Support as `repository`. |
| GitHub shorthand | `eamars/KazusaAIChatbot` | Support through `repo_hint`. |
| Markdown link in question | `[repo](https://github.com/owner/repo)` | Support after URL extraction. |
| GitHub tree URL | `/tree/main/src/foo` | Support as `directory`. |
| GitHub blob URL | `/blob/main/src/foo.py` | Support as `file`. |
| GitHub raw URL | `raw.githubusercontent.com/owner/repo/ref/path` | Support as `file`. |
| Explicit ref | branch, tag, or commit SHA | Support through URL/ref parsing and metadata. |
| Explicit local checkout | `local_root_hint` points at a git worktree | Support read-only. |
| Explicit local path inside checkout | `local_path_hint` points inside a git worktree | Support as `file` or `directory`. |
| Multiple same-repo candidates | repo plus one blob/tree URL | Support the most specific scope. |
| Multiple different repos | two unrelated repo URLs | Return `needs_user_input`. |
| No source | question has no URL or hint | Return `needs_user_input`. |
| GitHub issue, PR, or discussion | `/issues/1`, `/pull/2` | Return `rejected` with unsupported-source limitation. |
| GitHub archive/release asset | `.zip`, `.tar.gz`, `/releases` | Return `rejected` with unsupported-source limitation. |
| SSH or git protocol URL | `git@github.com:owner/repo.git` | Return `rejected`. |
| Credential-bearing URL | `https://token@github.com/owner/repo` | Return `rejected` and redact credential text. |
| Non-GitHub provider URL | GitLab, Bitbucket, Azure DevOps | Return `rejected`. |
| Package registry name | npm, PyPI, Cargo, Go module | Return `rejected`. |
| Generic paste/raw URL | Gist, Pastebin, arbitrary raw HTTP | Return `rejected`. |
| Pasted source text | code block in user text | Return `needs_user_input`; not a fetching source. |
| Path traversal or unsafe path | `../../`, `.git/config` | Return `rejected`. |

## Contracts And Data Shapes

### Public Entrypoint

```python
async def run(request: CodeFetchingRequest) -> CodeFetchingResult:
    ...
```

`src/kazusa_ai_chatbot/coding_agent/code_fetching/__init__.py` exposes only
`run` and public TypedDict contracts.

### Request

```python
class CodeFetchingRequest(TypedDict, total=False):
    question: str
    source_url: str
    repo_url: str
    repo_hint: str
    local_root_hint: str
    local_path_hint: str
    requested_ref: str
    source_scope_hint: Literal["repository", "directory", "file"]
    workspace_root: str
```

At least one repository source is required: `repo_url`, `repo_hint`,
`source_url`, `local_root_hint`, `local_path_hint`, or a public GitHub URL
embedded in `question`.

### Source Scope

```python
class CodeSourceScope(TypedDict):
    kind: Literal["repository", "directory", "file"]
    repo_relative_path: str | None
    source_url: str
    requested_ref: str | None
    interpretation: str
```

### Repository Ref

```python
class CodeRepositoryRef(TypedDict):
    provider: Literal["github"]
    owner: str
    repo: str
    source_url: str
    requested_ref: str | None
    resolved_ref: str
    current_commit: str
    default_branch: str
    local_root: str
    storage_kind: Literal["existing_local_checkout", "managed_clone"]
    managed_checkout: bool
    workspace_root: str | None
    cache_key: str | None
    dirty_state: Literal["clean", "dirty", "unknown"]
```

### Result

```python
class CodeFetchingResult(TypedDict):
    status: Literal["succeeded", "failed", "needs_user_input", "rejected"]
    message: str
    repository: CodeRepositoryRef | None
    source_scope: CodeSourceScope | None
    limitations: list[str]
    trace_summary: list[str]
```

Public traces must not expose secrets, raw command output, `.git` internals, or
absolute paths outside the resolved source when not needed for diagnostics.

## LLM Call And Context Budget

Before Phase 0:

- `code_fetching` does not exist.

After Phase 0:

- Kazusa live path: no new LLM calls.
- Kazusa background-worker path: no new LLM calls.
- Direct `code_fetching` path: no LLM calls.

If a later plan adds LLM source disambiguation, the prompt must receive only
question text, sanitized source candidate labels, URL kinds, and short
deterministic interpretations. It must not receive filesystem paths, command
strings, raw git output, or file content.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/README.md`: staged top-level ICD.
- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: package marker and future
  integration note.
- `src/kazusa_ai_chatbot/coding_agent/models.py`: shared Phase 0 source
  models and limits.
- `src/kazusa_ai_chatbot/coding_agent/tools/`: deterministic `git` metadata,
  path safety, and managed-workspace helpers required by fetching.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md`: subagent ICD.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/__init__.py`: public
  subagent entrypoint.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/models.py`: request,
  source-scope, repository, and result shapes.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py`: orchestration.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/github.py`: deterministic
  GitHub URL parsing.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_scope.py`: bounded
  deterministic source-scope routing.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/local_checkout.py`: local
  checkout matching and metadata.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_clone.py`: managed
  clone path and metadata handling.
- `tests/test_coding_agent_fetching.py`: direct subagent contract tests.
- `tests/test_coding_agent_fetching_internet.py`: explicit live-internet
  smoke for 10 public GitHub sources.

### Modify

- `development_plans/README.md`: add Phase 0 registry row.
- `development_plans/reference/designs/coding_agent_architecture.md`: align
  roadmap and contracts with Phase 0.
- `pytest.ini`: register and exclude the `live_internet` marker from default
  test runs.

### Keep

- `src/kazusa_ai_chatbot/background_work/*`: no Phase 0 changes.
- `src/kazusa_ai_chatbot/action_spec/*`: no Phase 0 changes.
- `src/kazusa_ai_chatbot/cognition_chain_core/*`: no Phase 0 changes.
- `src/kazusa_ai_chatbot/service.py`: no Phase 0 changes.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/`: not created in Phase 0.

## Overdesign Guardrail

- Actual problem: upstream coding-agent stages need a safe source-resolution
  and storage contract before any code reading can be reliable.
- Minimal change: add only the `code_fetching` package, README ICD, direct
  tests, deterministic URL/storage tooling, and deterministic source-scope
  routing.
- Ownership boundaries: `code_fetching` owns source resolution and storage;
  deterministic code owns safety, command execution, and source-kind
  classification.
- Rejected complexity: code reading, answer synthesis, supervisor loop,
  service integration, background-worker integration, code writing, execution,
  private repos, authenticated repos, LLM source routing, and web-agent help.
- Evidence threshold: add reading only after Phase 0 returns stable
  `CodeFetchingResult` objects for repo, tree, and file inputs.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan and the `code_fetching` README ICD.
- The responsible agent must not introduce new integrations, fallback paths, or
  extra capabilities.
- The responsible agent must treat changes outside `coding_agent`,
  listed tests, this plan, the registry, and the architecture reference as out
  of scope.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent creates focused tests for `CodeFetchingRequest`, `CodeFetchingResult`,
   GitHub URL parsing, unsupported source handling, storage paths, and metadata
   mismatch refusal.
2. Parent records expected failures before production-code edits.
3. Parent implements package scaffolding, README ICD, tools,
   deterministic parsing, source-scope routing, local checkout matching, and
   managed clone storage.
4. Parent runs focused tests.
5. Parent runs static greps proving no runtime integration changed.
6. Parent runs the live-internet 10-source smoke.
7. Parent performs fallback independent code review from a fresh-review stance
   and remediates findings in scope.

## Execution Model

- Parent agent owns orchestration, tests, verification, execution evidence,
  review remediation, lifecycle updates, and final sign-off.
- Parent establishes focused tests first and records expected failures.
- User-approved fallback execution replaces the normal production-code
  subagent for this plan. The parent agent owns planned production code
  changes directly.
- User-approved fallback review replaces the normal independent code-review
  subagent for this plan. After verification passes, the parent rereads the
  plan and diff from a fresh-review posture, records findings, and fixes only
  in-scope issues.

## Progress Checklist

- [x] Stage 1 - fetching contract tests established
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q`
  - Expected before implementation: missing module or missing entrypoint.
  - Evidence: record failure output.
  - Sign-off: `Codex/2026-06-20`.

- [x] Stage 2 - code_fetching package implemented
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q`
  - Evidence: record changed files and passing output.
  - Sign-off: `Codex/2026-06-20`.

- [x] Stage 3 - static boundary checks complete
  - Verify:
    `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
    returns no matches.
  - Verify: `git diff --check`.
  - Evidence: record grep and whitespace output.
  - Sign-off: `Codex/2026-06-20`.

- [x] Stage 4 - live internet source smoke checked
  - Verify:
    `venv\Scripts\python.exe -m pytest -m live_internet tests/test_coding_agent_fetching_internet.py -q -s`
  - Evidence: record the 10 source URLs and resolved statuses.
  - Sign-off: `Codex/2026-06-20`.

- [x] Stage 5 - fallback independent code review complete
  - Verify: rerun affected focused tests after review fixes.
  - Evidence: record findings, fixes, residual risks, and approval status.
  - Sign-off: `Codex/2026-06-20`.

## Verification

### Static Greps

- `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
  - Expected: no matches.
- `rg "code_reading|answer_code_question" src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_fetching.py`
  - Expected: matches only in README future-stage notes.

### Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q
```

### Live Internet Smoke

```powershell
venv\Scripts\python.exe -m pytest -m live_internet tests/test_coding_agent_fetching_internet.py -q -s
```

This gate must exercise 10 public GitHub source forms through
`code_fetching.run(...)`. If network access is blocked by the local sandbox,
request escalation and rerun the exact command.

### Static Checks

```powershell
git diff --check
```

## Independent Plan Review

Before approval, review this plan against the architecture reference and the
Phase 1 reading plan. The reviewer must confirm Phase 0 owns only fetching,
storage, source-scope routing, and its README ICD, and that Phase 1 can consume
`CodeFetchingResult` without importing private fetching internals.

### 2026-06-20 review outcome

Findings resolved in this revision:

- Blocker: the draft did not enumerate all expected input classes. Resolved by
  adding `Supported And Unsupported Input Matrix` and matching test criteria.
- Blocker: the draft allowed implicit temp workspace use without a production
  configuration boundary. Resolved by requiring future worker integration to
  pass an explicit workspace root while keeping temp fallback for standalone
  direct use.
- Blocker: the draft execution model still required subagents. Resolved by
  recording the user's explicit no-subagent fallback approval for this plan.
- Blocker: the draft verification did not include the requested 10 public
  internet code sources. Resolved by adding the live-internet smoke gate.
- Non-blocking finding: Phase 0 LLM source routing created prompt/test
  complexity without a deterministic failure requiring it. Resolved by making
  Phase 0 source routing deterministic and deferring LLM disambiguation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Because the user explicitly requested no subagent execution, the parent agent
must perform this review from a fresh-review posture. The review must inspect
the plan, diff, README ICD, focused tests, storage safety, path containment,
and absence of service integration.

The parent fixes findings only inside the approved change surface. If a finding
requires reading, service integration, or contract expansion, stop and update a
plan before changing code.

## Acceptance Criteria

This plan is complete when:

- `code_fetching` exists as a package with README ICD.
- `code_fetching.run(...)` returns `CodeFetchingResult`.
- Repo, tree, blob, raw-file, embedded question URL, local checkout, and
  ambiguous-source cases are covered by deterministic tests.
- Unsupported source classes from the input matrix are covered by deterministic
  tests.
- Managed clones use the documented workspace layout and metadata guard.
- Existing local checkouts are not mutated.
- No code-reading or answer-synthesis path exists.
- Static greps, focused tests, live-internet 10-source smoke,
  `git diff --check`, and fallback independent code review pass or have
  user-approved exceptions recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Fetched code pollutes active checkout | Default managed storage uses temp-dir workspace outside the repo. | Storage path tests. |
| Metadata mismatch overwrites unrelated clone | Require matching `metadata.json`; reject mismatch. | Managed clone tests. |
| Unsupported source is guessed incorrectly | Unsupported classes return `rejected` or `needs_user_input`. | Input-matrix tests. |
| Phase 0 expands into reading | Deferred scope and grep for `code_reading`. | Static grep and code review. |

## Execution Evidence

### 2026-06-20 draft

- User split coding-agent staging:
  - Phase 0: `code_fetching`;
  - Phase 1: `code_reading`.
- User required each subagent to have a README ICD.
- User required fetched-code storage to be explicit.
- Independent plan review surfaced input-matrix, workspace-config,
  no-subagent execution, and 10-source verification gaps; this revision fixed
  those gaps.
- User approved no-subagent implementation for this plan.

### 2026-06-20 execution

- Stage 1 expected failure recorded:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py -q`
  failed during collection with
  `ModuleNotFoundError: No module named 'kazusa_ai_chatbot.coding_agent'`.
- Stage 2 focused verification passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py -q`
  initially reported `12 passed in 0.88s`; after fallback code-review fixes,
  the same focused suite reported `15 passed in 1.09s`.
- Stage 3 static checks passed:
  `rg "coding_agent" src\kazusa_ai_chatbot\background_work src\kazusa_ai_chatbot\action_spec src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\service.py`
  returned no matches; `rg "code_reading|answer_code_question" ...` returned
  only `src\kazusa_ai_chatbot\coding_agent\README.md`; `py_compile` passed for
  new Python files and tests; `git diff --check` reported no whitespace errors.
- Stage 4 live-internet smoke passed:
  `venv\Scripts\python -m pytest -m live_internet tests\test_coding_agent_fetching_internet.py -q -s`
  reported `1 passed in 9.61s` for:
  `octocat/Hello-World`, `octocat/Spoon-Knife`, `github/gitignore`,
  `github/gitignore/tree/main/Global`, `github/gitignore/blob/main/Python.gitignore`,
  `raw.githubusercontent.com/github/gitignore/main/Node.gitignore`,
  `pypa/sampleproject`, `pallets/itsdangerous`, `pallets/markupsafe`, and
  `pallets/click`.
- Stage 5 fallback independent code review completed:
  - Finding: the source selector used a public `CodeFetchingResult` repository
    marker as an internal transport and required `type: ignore` reparsing.
    Fix: replaced that transport with private `_SourceSelection`.
  - Finding: repository-internal `.git` paths were not explicitly rejected as
    source scopes. Fix: added GitHub URL and local checkout path guards plus
    deterministic tests for unsafe URL and local `.git/config` inputs.
  - Residual risk accepted for Phase 0: managed checkout reuse is clone-or-use
    and does not auto-pull existing managed clones. This matches the Phase 0
    storage boundary; refresh policy should be specified before worker
    integration.
  - Review verification:
    `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py -q`
    reported `15 passed in 1.09s`;
    `venv\Scripts\python -m pytest -m live_internet tests\test_coding_agent_fetching_internet.py -q -s`
    reported `1 passed in 9.91s`;
    `py_compile` passed for the new package and tests; static greps confirmed
    no service/core integration and no `type: ignore` or broad exception
    findings; `git diff --check` reported no whitespace errors, only existing
    CRLF normalization warnings for edited text files.
