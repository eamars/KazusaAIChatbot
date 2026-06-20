# coding_agent phase 1 reading plan

## Summary

- Goal: Add the standalone `code_reading` subagent package and top-level
  direct answer interface on top of the completed Phase 0 `code_fetching`
  contract.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: additive standalone module only; no Kazusa service,
  L2d, background-work, result-ready, or placeholder-removal integration in
  Phase 1.
- Highest-risk areas: unbounded source prompts, hallucinated code
  architecture, leaking raw source or absolute paths, bypassing the Phase 0
  fetching ICD, incomplete Phase 2 worker handoff, and failing the target
  image-reading question.
- Acceptance criteria: direct tests can call `answer_code_question(...)`, use
  Phase 0 `CodeFetchingResult`, classify read-only code questions against the
  Phase 1 query taxonomy, read bounded source evidence, and return
  evidence-backed answers, clarification requests, or explicit rejections with
  public-safe metadata.

## Context

Phase 0 owns source resolution and managed storage through
`code_fetching.run(...)`. Phase 1 must not reimplement fetching. It consumes the
Phase 0 result and adds code reading, evidence assembly, answer synthesis, and
the top-level direct coding-agent interface.

The target real-world question remains:

```text
[eamars/KazusaAIChatbot](https://github.com/eamars/KazusaAIChatbot) 项目是怎么实现读图的
```

Phase 1 implements this direct path:

```text
CodingAgentRequest
  -> coding_agent.answer_code_question
  -> code_fetching.run from Phase 0
  -> code_reading.run
  -> CodingAgentResponse
```

Phase 1 also leaves a concrete Phase 2 handoff contract. It does not register
the worker or touch the background-work runtime, but it defines how the direct
`CodingAgentResponse` maps into the existing `BackgroundWorkResult` shape so
the next plan does not have to invent the result boundary.

Each subagent has an upstream ICD:

- `coding_agent/README.md`: external direct-interface ICD;
- `code_fetching/README.md`: supervisor/upstream to fetching ICD from Phase 0;
- `code_reading/README.md`: supervisor/upstream to reading ICD added here.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing reading prompts, answer
  synthesis, LLM call counts, context budgets, or subagent boundaries.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python tests or source containing the
  Chinese image-reading acceptance question.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running live LLM code-reading checks.

## Mandatory Rules

- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval and status `approved` or
  `in_progress`.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use parent-led native subagent execution. If native subagent capability is
  unavailable, stop before production implementation unless the user explicitly
  approves fallback execution.
- Use `venv\Scripts\python.exe` for Python commands.
- Never read `.env` during implementation or verification.
- Treat Phase 0 as a prerequisite. Do not bypass `code_fetching.run(...)` or
  import private fetching modules.
- The top-level supervisor consumes `CodeFetchingResult`. `code_reading`
  consumes only the successful `CodeRepositoryRef` and `CodeSourceScope`.
- When Phase 0 fetching returns `failed`, `rejected`, or `needs_user_input`,
  return the same status without calling `code_reading`.
- Do not expose raw `CodeRepositoryRef` in public `CodingAgentResponse`
  fields. Public responses and Phase 2 handoff metadata must use a sanitized
  repository summary that excludes `local_root`, `workspace_root`, and
  `cache_key`.
- Carry Phase 0 limitations into the final response, including successful
  mixed-input limitations and dirty-local-checkout warnings.
- Keep Phase 1 independent from Kazusa runtime integration. Do not modify L2d,
  action-spec, background-work, result-ready cognition, service delivery,
  adapters, persistence, consolidation, or scheduler code.
- Keep `code_reading` as a subagent package with its own
  `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md` ICD.
- Full source files must not enter prompts by default; only bounded evidence
  rows may enter answer synthesis.

## Must Do

- Update `src/kazusa_ai_chatbot/coding_agent/README.md` with the Phase 1 direct
  answer interface and its dependency on Phase 0 fetching.
- Add `answer_code_question(request: CodingAgentRequest) -> CodingAgentResponse`
  as the external direct coding-agent entrypoint.
- Add a minimal top-level supervisor/orchestrator that calls Phase 0
  `code_fetching.run(...)`, then `code_reading.run(...)`, then returns the
  final response. The supervisor must pass through all public Phase 0 source
  fields instead of narrowing the fetching contract.
- Add a public-safe repository projection for direct responses and future
  background-work metadata.
- Add deterministic supervisor behavior for non-success Phase 0 results: no
  reading call, same status, empty answer/evidence, and carried limitations.
- Add dirty existing-local-checkout handling: allow reading but append a
  limitation stating that evidence may reflect uncommitted local changes.
- Create `src/kazusa_ai_chatbot/coding_agent/code_reading/` as a subagent
  package.
- Add `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md` as the
  upstream ICD for the reading subagent.
- Add `code_reading.run(request: CodeReadingRequest) -> CodeReadingResult`.
- Implement reading over Phase 0 `CodeRepositoryRef` and `CodeSourceScope`
  using `rg --files`, `rg -n --json`, and bounded file reads.
- Implement strict source-scope containment: repository scope may inspect the
  repo, directory scope may inspect only files under `repo_relative_path`, and
  file scope may inspect only that file unless a bounded repo-level symbol
  lookup is explicitly recorded as supporting context.
- Implement read guards for `.git`, `.env`, secret-like files, binary files,
  and paths outside `repository.local_root`.
- Implement evidence row assembly and answer synthesis.
- Define the Phase 2 handoff metadata contract in `coding_agent/README.md`,
  including worker name, worker description, `BackgroundWorkResult` mapping,
  and fields forbidden from worker metadata.
- Add deterministic tests for reading planner behavior, evidence assembly,
  source-scope handling, prompt budget caps, answer response shape, README ICD
  consistency, and the target image-reading answer.
- Add deterministic query-taxonomy tests covering supported reading questions,
  bounded partial-answer cases, clarification cases, and explicit rejection
  cases listed in `Phase 1 Query Taxonomy And Pass Criteria`.
- Add deterministic tests proving `answer_code_question(...)` passes all
  public Phase 0 source fields through to `code_fetching.run(...)`.
- Add deterministic tests proving question-only input with an embedded GitHub
  URL works, because Phase 2 background-work jobs provide a task brief rather
  than worker-local repo parameters.
- Add one live LLM smoke test for direct `answer_code_question(...)` when live
  LLM configuration is available.

## Deferred

- Do not implement or rewrite Phase 0 fetching internals except for narrow ICD
  import fixes required by this plan.
- Do not connect `coding_agent` to Kazusa background-work worker registry.
- Do not remove or decommission `text_artifact`.
- Do not update L2d, action-spec affordance wording, background-work router,
  provider dispatch, result-ready cognition, L3/dialog, service delivery,
  adapters, persistence, consolidation, or scheduler code.
- Do not implement code writing, patch proposal, patch apply, code execution,
  Docker execution, package installation, arbitrary shell, private repos,
  authenticated repos, or `web_agent3` calls.

## Cutover Policy

Overall strategy: additive standalone module. Phase 1 has no Kazusa runtime
cutover.

| Area | Policy | Instruction |
|---|---|---|
| Phase 0 fetching | compatible prerequisite | Consume only its public ICD and `CodeFetchingResult`. |
| `code_reading` subagent | additive | Implement package, README ICD, direct tests, and public subagent entrypoint. |
| Top-level direct interface | additive | Add `answer_code_question(...)` for tests and future integration. |
| Kazusa runtime paths | no-op | Do not wire into service, L2d, background work, or result-ready delivery. |
| Existing workers | no-op | Keep production behavior unchanged. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not expand Phase 1 into service integration or code writing.
- For no-op areas, leave existing runtime behavior unchanged.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

After Phase 1, direct callers can ask a repository/source-code question through
`kazusa_ai_chatbot.coding_agent.answer_code_question(...)`. The implementation
uses Phase 0 fetching to resolve the repository and source scope, then uses
`code_reading` to gather bounded evidence and synthesize an answer.

The direct acceptance call for the target question returns a Chinese answer
explaining these facts with file evidence:

- adapters normalize image attachments and may provide inline `base64_data`;
- service builds `user_multimedia_input`;
- the graph routes multimedia turns through `multimedia_descriptor_agent`
  before relevance;
- `VISION_DESCRIPTOR_LLM` converts the image payload into `description` and
  structured `image_observation`;
- media descriptors are cached and persisted back to attachment descriptions;
- `cognition_episode` and prompt selection project `image_observation` as a
  typed media percept instead of treating image descriptions as authored user
  text;
- history/reply/RAG use stored attachment descriptions or `<image>...</image>`
  projection.

## Phase 1 Query Taxonomy And Pass Criteria

Phase 1 robustness means every direct request reaches one of three outcomes:
answer from bounded local evidence, ask for clarification, or reject as outside
read-only scope. It does not mean every code task is supported.

### Supported Read-Only Query Families

Each supported family must have at least one deterministic fixture test that
checks answer status, evidence paths, limitations, and public-safe metadata.

| Family | Example request shape | Pass criterion |
|---|---|---|
| Feature or pipeline explanation | "How does this project read images?" | Explains end-to-end flow with source evidence from multiple relevant files. |
| Architecture or module responsibility | "What owns background work routing?" | Identifies modules, responsibilities, and boundaries without inventing missing layers. |
| Public API, route, CLI, or contract lookup | "What is the request shape for this endpoint?" | Returns concrete signatures, schema fields, or command names with evidence. |
| Symbol, class, or function explanation | "What does `Foo.run` do?" | Locates definition and summarizes inputs, outputs, side effects, and callers when visible. |
| Definition and usage search | "Where is `VISION_DESCRIPTOR_LLM` used?" | Lists definition and representative usages with repo-relative file evidence. |
| File, package, or directory summary | "Summarize this folder." | Summarizes responsibilities from bounded files under the requested scope. |
| Data, config, prompt, or state model reading | "How is this config/state field used?" | Tracks field creation, validation, and consumption without reading `.env`. |
| Error, lifecycle, cache, or persistence path | "How are failures cached or stored?" | Explains the control flow and names uncertainty when evidence is incomplete. |
| Test coverage mapping | "What tests cover image handling?" | Maps behavior to test files and states apparent uncovered areas as limitations. |
| Dependency or external integration usage | "How does it call Mongo/OpenAI/FastAPI?" | Uses imports, wrappers, settings, and call sites; does not claim current external facts. |
| Intra-repo comparison | "Compare these two handlers." | Compares responsibilities and behavior using evidence from both sides. |
| Docs-to-code consistency | "Does README match implementation?" | Compares docs and source, reporting only evidence-backed mismatches. |
| Static impact or risk read | "What might depend on this module?" | Gives evidence-backed dependency and caller notes without proposing patches. |
| Build, run, or deployment reading | "How do I run this project?" | Summarizes checked-in docs/scripts only; does not execute commands. |

### Bounded Or Partial Outcomes

- Ambiguous symbol, feature name, source scope, or repository target returns
  `needs_user_input` with a concise clarification question.
- Very broad questions may answer with the top evidence-backed architecture and
  include a limitation, or ask for narrowing when evidence selection would be
  arbitrary.
- File and directory scoped questions must honor Phase 0 source scope; any
  repo-wide lookup used as supporting context must be named in `trace_summary`.
- Incomplete evidence must produce an answer with explicit uncertainty instead
  of filling gaps from general programming knowledge.
- The answer language follows `preferred_language` or the user's question when
  practical; code identifiers and quoted evidence stay exact.

### Explicit Rejections

Phase 1 returns `rejected` for requests to write or apply code changes, propose
patches as the primary artifact, run tests or shell commands, install packages,
inspect secrets or `.env`, dump large raw files, analyze binary assets, use
private/authenticated sources unsupported by Phase 0, certify legal/security
status, or answer current external dependency facts that require web evidence.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stage dependency | Phase 1 depends on Phase 0 fetching. | Reading should consume a stable source contract instead of resolving storage itself. |
| Subagent packaging | `code_reading` is a folder with README ICD. | It is an owned subagent boundary. |
| Top-level interface | Add `answer_code_question(...)` in Phase 1. | The direct public answer API is meaningful only once reading exists. |
| Evidence | Synthesize only from bounded evidence rows. | Prevents unbounded source prompt leakage. |
| Source scope | Reading honors Phase 0 `repository`, `directory`, or `file` scope. | File and tree URLs should focus reading without losing repo context. |
| Fetching handoff | Phase 0 validates scoped GitHub paths and returns public-safe local source labels. | Reading can trust `repository.local_root` for filesystem access while keeping public metadata sanitized. |
| Fetch failure handling | Supervisor short-circuits on non-success Phase 0 results. | Reading should never receive `None` repository or source scope. |
| Public repository metadata | Use `CodingAgentRepositorySummary` in public responses. | Avoids leaking absolute local paths or managed workspace internals. |
| Dirty local checkout | Allow existing local dirty checkouts and add a limitation. | Direct local users may intentionally ask about worktree changes; the answer must disclose that evidence may include them. |
| Phase 2 handoff | Define `WORKER="coding_agent"` and a `BackgroundWorkResult` mapping, but do not register it. | Leaves the next integration plan a concrete contract without touching runtime code now. |
| Query taxonomy | Support broad read-only repository questions through answer, clarification, or rejection outcomes. | A robust reading agent needs explicit pass criteria beyond the single image-reading acceptance case. |
| LLM route | Use existing background-work-route LLM configuration for direct tests. | Avoids adding runtime config before service integration. |

## Contracts And Data Shapes

### Public Entrypoint

```python
async def answer_code_question(
    request: CodingAgentRequest,
) -> CodingAgentResponse:
    ...
```

### Request

```python
class CodingAgentRequest(TypedDict, total=False):
    question: str
    source_url: str
    repo_url: str
    repo_hint: str
    local_root_hint: str
    local_path_hint: str
    requested_ref: str
    source_scope_hint: Literal["repository", "directory", "file"]
    workspace_root: str
    preferred_language: str
    max_answer_chars: int
```

At least one repository source is required by Phase 0 fetching:
`source_url`, `repo_url`, `repo_hint`, `local_root_hint`,
`local_path_hint`, or a public GitHub URL embedded in `question`.

### Code Reading Request

```python
class CodeReadingRequest(TypedDict):
    question: str
    repository: CodeRepositoryRef
    source_scope: CodeSourceScope
    preferred_language: str
    max_answer_chars: int
```

`CodeReadingRequest` is constructed only after Phase 0 returns
`status="succeeded"` with non-null `repository` and `source_scope`.

### Public Repository Summary

```python
class CodingAgentRepositorySummary(TypedDict):
    provider: Literal["github"]
    owner: str
    repo: str
    source_url: str
    requested_ref: str | None
    resolved_ref: str
    current_commit: str
    default_branch: str
    storage_kind: Literal["existing_local_checkout", "managed_clone"]
    managed_checkout: bool
    dirty_state: Literal["clean", "dirty", "unknown"]
```

This is the only repository shape allowed in `CodingAgentResponse` and future
background-worker metadata. It intentionally excludes `local_root`,
`workspace_root`, and `cache_key`. `code_reading` may use the full
`CodeRepositoryRef` internally for filesystem reads, but public outputs must
use this projection.

### Evidence Row

```python
class CodeEvidenceRow(TypedDict):
    path: str
    line_start: int | None
    line_end: int | None
    symbol_or_topic: str
    excerpt: str
    reason: str
```

### Reading Result

```python
class CodeReadingResult(TypedDict):
    status: Literal["succeeded", "failed", "needs_user_input", "rejected"]
    answer_text: str
    evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]
```

### Coding Agent Response

```python
class CodingAgentResponse(TypedDict):
    status: Literal["succeeded", "failed", "needs_user_input", "rejected"]
    answer_text: str
    repository: CodingAgentRepositorySummary | None
    source_scope: CodeSourceScope | None
    evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]
```

Public response fields must contain repo-relative evidence paths only. They
must not include raw command output, raw source dumps, secrets, adapter IDs,
job IDs, leases, retries, or absolute paths outside diagnostics.

### Supervisor Result Rules

- If `code_fetching.run(...)` returns `failed`, `rejected`, or
  `needs_user_input`, `answer_code_question(...)` returns that status, empty
  `answer_text`, `repository=None`, `source_scope=None`, empty evidence,
  carried `limitations`, and a trace summary that includes the fetching trace.
- If fetching succeeds, the supervisor builds `CodeReadingRequest` from the
  full internal `CodeRepositoryRef` and `CodeSourceScope`.
- If the fetched repository has `dirty_state="dirty"`, the supervisor appends a
  limitation before reading: `Existing local checkout is dirty; evidence may
  include uncommitted local changes.`
- The final response limitations are the ordered union of fetching
  limitations, supervisor limitations, and reading limitations.

### Phase 2 Worker Handoff Contract

Phase 1 does not register a background-work worker, but it defines the worker
surface Phase 2 must use:

```python
WORKER = "coding_agent"
DESCRIPTION = (
    "Answers repository and codebase questions with bounded local source "
    "evidence; rejects code writing, code execution, patching, and package "
    "installation."
)
```

Phase 2 maps one `CodingAgentResponse` into the existing
`BackgroundWorkResult` contract:

```python
class CodingAgentWorkerMetadata(TypedDict, total=False):
    task_type: Literal["code_question"]
    repository: CodingAgentRepositorySummary
    source_scope: CodeSourceScope
    evidence: list[CodeEvidenceReference]
    limitations: list[str]


class CodeEvidenceReference(TypedDict):
    path: str
    line_start: int | None
    line_end: int | None
    symbol_or_topic: str
```

Mapping rules:

- The Phase 2 worker builds `CodingAgentRequest.question` from
  `BackgroundWorkWorkerDecision.source_summary` when present. The existing
  background-work worker loop sets `source_summary` from `source_context` or
  falls back to the queued `task_brief`.
- The Phase 2 worker supplies `workspace_root` from a configured coding-agent
  workspace setting. It must not parse a workspace path from user text and must
  not use Phase 0's standalone temp fallback.
- `BackgroundWorkResult.status` equals `CodingAgentResponse.status`.
- `BackgroundWorkResult.worker` is `coding_agent`.
- `artifact_text` is `answer_text` capped by the job `max_output_chars`.
- `result_summary` is a short status summary with repo, commit, and evidence
  count when succeeded.
- `failure_summary` is empty on success; otherwise it is the most specific
  limitation or status message.
- `worker_metadata.repository` uses `CodingAgentRepositorySummary`.
- `worker_metadata.evidence` excludes excerpts and includes only repo-relative
  path, line range, and topic.
- Worker metadata must not contain `local_root`, `workspace_root`, `cache_key`,
  raw source excerpts, raw command output, job ids, leases, adapter ids, or
  platform delivery fields.

## LLM Call And Context Budget

Before Phase 1:

- Phase 0 source resolution uses no LLM calls.
- No code-reading LLM calls exist.

After Phase 1:

- Kazusa live path: no new LLM calls.
- Kazusa background-worker path: no new LLM calls.
- Direct `answer_code_question(...)` path:
  - Phase 0 source resolution: no LLM calls;
  - code reading planner/synthesizer: up to two LLM calls;
  - no `web_agent3` calls.

Context budget:

- Use `50k characters` as the maximum coding-agent prompt budget unless a
  lower model limit is configured.
- The planner's semantic question is limited to: identify the read-only query
  family, required evidence topics, and whether the request should be answered,
  clarified, or rejected.
- The LLM prompt receives repository identity, source scope, question, and
  bounded evidence rows only. It must not receive `local_root`,
  `workspace_root`, `cache_key`, raw command output, or full source files.
- Each file excerpt is capped before entering prompts.
- Search results are capped by match count and line length.
- Full source files are never inserted into prompts by default.
- Final `answer_text` is capped by `max_answer_chars`.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md`: reading subagent
  ICD.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/__init__.py`: public
  subagent entrypoint.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/models.py`: request,
  result, and evidence contracts.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/agent.py`: reading
  orchestration.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/planner.py`: bounded search
  planning.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/evidence.py`: evidence row
  assembly.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/synthesizer.py`: answer
  synthesis.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/prompts.py`: prompt
  constants.
- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: direct answer
  orchestration over fetching and reading.
- `tests/test_coding_agent_reading.py`: direct reading subagent tests.
- `tests/test_coding_agent_interface.py`: top-level answer response tests.
- `tests/test_coding_agent_image_reading_acceptance.py`: target question
  acceptance test.
- `tests/test_coding_agent_live_llm.py`: conditional live direct-answer smoke.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/README.md`: add Phase 1 answer ICD.
  Include the public-safe repository summary and Phase 2 worker handoff
  contract.
- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: export
  `answer_code_question`.
- `src/kazusa_ai_chatbot/coding_agent/models.py`: add reading and response
  contracts if not already present.
- `development_plans/README.md`: add Phase 1 registry row.
- `development_plans/reference/designs/coding_agent_architecture.md`: align
  roadmap and real-demand mapping.

### Keep

- `src/kazusa_ai_chatbot/coding_agent/code_fetching/*`: consume public ICD
  only; do not rewrite internals.
- `src/kazusa_ai_chatbot/background_work/*`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/action_spec/*`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/cognition_chain_core/*`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/service.py`: no Phase 1 changes.

## Overdesign Guardrail

- Actual problem: after Phase 0 can resolve source, the coding agent still
  cannot read code and answer repository questions.
- Minimal change: add only `code_reading`, its README ICD, evidence assembly,
  answer synthesis, and the direct top-level answer interface.
- Ownership boundaries: Phase 0 owns source resolution and storage;
  `code_reading` owns evidence and synthesis; tools own path safety and bounded
  reads; no Kazusa runtime layer is touched.
- Rejected complexity: fetching rewrites, service integration, background
  worker registration, placeholder removal, code writing, execution, patching,
  private repos, authenticated repos, and web-agent help.
- Evidence threshold: add service integration only after direct Phase 1
  acceptance passes and a later approved plan names the integration target.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan and the README ICDs.
- The responsible agent must not introduce new integrations, fallback paths, or
  extra capabilities.
- The responsible agent must treat changes outside `coding_agent`, listed
  tests, this plan, registry, and architecture reference as out of scope.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent verifies Phase 0 completion or records Phase 0 as a blocker.
2. Parent creates focused tests for `CodeReadingRequest`, `CodeReadingResult`,
   evidence rows, source-scope behavior, public-safe repository projection,
   Phase 0 non-success short-circuiting, Phase 0 limitation carryover, dirty
   local checkout warnings, query-taxonomy pass criteria, all Phase 0
   source-field pass-through, and `answer_code_question`.
3. Parent records expected failures before production-code edits.
4. Production-code subagent implements reading package, README ICD, supervisor,
   and direct answer entrypoint.
5. Parent runs focused reading and interface tests.
6. Parent runs target image-reading acceptance using question-only input with
   the embedded GitHub URL and a configured `workspace_root`.
7. Parent runs static greps proving no runtime integration changed.
8. Parent runs live LLM direct-answer smoke when available.
9. Parent starts independent code review and remediates findings in scope.

## Execution Model

- Parent agent owns orchestration, tests, verification, execution evidence,
  review remediation, lifecycle updates, and final sign-off.
- Parent establishes focused tests first and records expected failures.
- Production-code subagent owns planned production code changes only.
- Independent code-review subagent reviews after planned verification passes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 0 - Phase 0 prerequisite verified
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q`
  - Evidence: record pass output or Phase 0 blocker.
  - Sign-off: `Codex / 2026-06-20`.

- [x] Stage 1 - reading contract tests established
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_reading.py tests/test_coding_agent_interface.py -q`
  - Expected before implementation: missing module or missing entrypoint.
  - Must cover: source-scope containment, safe file filtering, public-safe
    repository summary, fetching non-success short-circuit, limitation
    carryover, dirty-local-checkout limitation, supported query families,
    partial-answer cases, clarification cases, explicit rejections, and all
    Phase 0 source-field pass-through.
  - Evidence: record failure output.
  - Sign-off: `Codex / 2026-06-20`.

- [x] Stage 2 - code_reading and direct answer implemented
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_reading.py tests/test_coding_agent_interface.py -q`
  - Evidence: record changed files and passing output.
  - Sign-off: `Codex / 2026-06-20`.

- [x] Stage 3 - real demand acceptance complete
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_image_reading_acceptance.py -q`
  - Expected input shape: question-only request containing the GitHub markdown
    link plus configured `workspace_root`.
  - Evidence: record generated answer excerpt, public-safe repository summary,
    source scope, and evidence list.
  - Sign-off: `Codex / 2026-06-20`.

- [x] Stage 4 - static boundary checks complete
  - Verify:
    `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
    returns no matches.
  - Verify: `git diff --check`.
  - Evidence: record grep and whitespace output.
  - Sign-off: `Codex / 2026-06-20`.

- [x] Stage 5 - live direct-answer smoke checked
  - Verify:
    `venv\Scripts\python.exe -m pytest -m live_llm tests/test_coding_agent_live_llm.py -q -s -k image_reading_question`
  - Completion rule: this is a conditional diagnostic. If live LLM
    configuration is unavailable, record the config blocker, mark this stage as
    checked with `skipped_unavailable`, and do not claim live LLM quality.
  - Evidence: record response status, evidence count, missing-fact notes, or
    unavailable-config blocker.
  - Sign-off: `Codex / 2026-06-20`.

- [x] Stage 6 - independent code review complete
  - Verify: rerun affected focused tests after review fixes.
  - Evidence: record findings, fixes, residual risks, and approval status.
  - Sign-off: `Codex / 2026-06-20`.

## Verification

### Static Greps

- `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
  - Expected: no matches.
- `rg "git clone|parse_github_source|parse_repo_hint|repo_url|local_root_hint|local_path_hint|requested_ref|source_scope_hint" src/kazusa_ai_chatbot/coding_agent/code_reading`
  - Expected: no matches outside README explanations; reading must consume
    Phase 0 output, not fetch.
- `rg "local_root|workspace_root|cache_key" src/kazusa_ai_chatbot/coding_agent/code_reading src/kazusa_ai_chatbot/coding_agent/supervisor.py tests/test_coding_agent_interface.py`
  - Expected: production code may read these fields internally; public response
    tests must assert they are absent from response repository and worker
    handoff metadata.

### Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_reading.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_interface.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_image_reading_acceptance.py -q
```

### Live LLM Smoke

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests/test_coding_agent_live_llm.py -q -s -k image_reading_question
```

If live LLM configuration is unavailable, record the blocker and mark Stage 5
as `skipped_unavailable`. This plan may still complete deterministic Phase 1
work, but final reporting must state that live LLM answer quality was not
verified.

### Static Checks

```powershell
git diff --check
```

## Independent Plan Review

Before approval, review this plan against the architecture reference and Phase
0 fetching plan. The reviewer must confirm Phase 1 consumes the Phase 0 ICD,
does not reimplement fetching, adds a `code_reading` README ICD, and keeps all
Kazusa runtime integration deferred. The reviewer must also confirm the Phase
2 handoff contract maps cleanly to existing `BackgroundWorkResult` without
absolute paths, worker-local action-spec fields, or adapter delivery fields.

### 2026-06-20 handoff review outcome

Findings resolved in this revision:

- Blocker: public `CodingAgentResponse.repository` exposed raw
  `CodeRepositoryRef` fields, including local filesystem roots. Resolved by
  adding `CodingAgentRepositorySummary` and requiring that projection in public
  response and Phase 2 metadata.
- Blocker: Phase 1 did not define the Phase 2 worker handoff. Resolved by
  defining `WORKER="coding_agent"`, a prompt-safe description, and
  `BackgroundWorkResult` mapping rules.
- Blocker: Phase 2 workspace ownership was not explicit. Resolved by requiring
  the future worker to supply a configured coding workspace root and forbidding
  user-text workspace parsing or temp fallback use.
- Blocker: fetch failure propagation was underspecified. Resolved by adding
  supervisor short-circuit rules for non-success `CodeFetchingResult`.
- Blocker: Phase 0 limitations were not carried forward. Resolved by requiring
  ordered limitation propagation from fetching, supervisor, and reading.
- Major finding: direct tests could pass only with structured repo fields while
  Phase 2 will pass a task brief. Resolved by requiring a question-only
  embedded-URL acceptance path.
- Major finding: dirty local checkout behavior was undefined. Resolved by
  allowing it with an explicit response limitation.
- Major finding: scoped reads, secret-like files, and binary files were not
  concrete enough. Resolved by adding containment and refusal requirements plus
  deterministic tests.
- Major finding: live LLM smoke was inconsistent as a completion gate. Resolved
  by making it a conditional diagnostic with explicit `skipped_unavailable`
  evidence semantics.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The review subagent must inspect the plan, diff, README ICDs, focused tests,
evidence safety, prompt safety, path containment, Phase 0 boundary adherence,
and absence of service integration. The review subagent reports findings only.

The parent fixes findings only inside the approved change surface. If a finding
requires fetching redesign, service integration, or contract expansion, stop
and update a plan before changing code.

## Acceptance Criteria

This plan is complete when:

- Phase 0 fetching tests pass or are recorded as a blocker before Phase 1 work.
- `code_reading` exists as a package with README ICD.
- `answer_code_question(...)` exists as the top-level direct answer entrypoint.
- The top-level supervisor consumes `CodeFetchingResult`; `code_reading`
  consumes only successful `CodeRepositoryRef` and `CodeSourceScope` instead
  of parsing URLs or cloning repos.
- `answer_code_question(...)` short-circuits Phase 0 non-success results
  without calling `code_reading`.
- Public responses use `CodingAgentRepositorySummary`; they do not expose
  `local_root`, `workspace_root`, or `cache_key`.
- `answer_code_question(...)` passes through all public Phase 0 source fields:
  `source_url`, `repo_url`, `repo_hint`, `local_root_hint`,
  `local_path_hint`, `requested_ref`, and `source_scope_hint`.
- Fetching limitations, dirty-local-checkout warnings, and reading limitations
  are carried into the final response.
- Supported query families in `Phase 1 Query Taxonomy And Pass Criteria` have
  deterministic answer-shape tests; bounded partial-answer, clarification, and
  rejection outcomes also have deterministic tests.
- Reading source-scope containment, `.git` refusal, `.env` refusal,
  secret-like file refusal, binary-file refusal, and path traversal refusal are
  covered by deterministic tests.
- Phase 2 `coding_agent` worker name, description, and `BackgroundWorkResult`
  mapping are documented in `coding_agent/README.md`.
- The target image-reading question is answered in Chinese with source-file
  evidence.
- No Phase 1 code path writes files in the active checkout, applies patches,
  installs packages, runs arbitrary shell commands, mutates the active
  checkout, or sends adapter text directly.
- Static greps, focused tests, `git diff --check`, and independent code review
  pass or have user-approved exceptions recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Reading reimplements fetching | Consume Phase 0 public ICD only. | Static grep and code review. |
| Source prompts are unbounded | Prompt only bounded evidence rows. | Reading tests and prompt review. |
| Hallucinated architecture | Require source evidence for answer topics. | Image-reading acceptance test. |
| Phase 1 expands into service integration | Mark runtime files as keep/no-op. | Static grep and diff review. |
| Public response leaks local paths | Use `CodingAgentRepositorySummary` and safe source labels. | Interface tests and static review. |
| Phase 2 cannot map direct response to worker result | Define worker name, description, and `BackgroundWorkResult` mapping now. | README ICD consistency test. |

## Execution Evidence

### 2026-06-20 draft

- User split coding-agent staging:
  - Phase 0: `code_fetching`;
  - Phase 1: `code_reading`.
- User required each subagent to have a README ICD.
- Phase 1 now depends on Phase 0 fetching instead of implementing fetching.
- Draft status: not approved for implementation.

### 2026-06-20 Phase 1 execution

- User approved execution with exactly one subagent. The single subagent edited
  tests only and reported coverage for all 14 query-taxonomy families,
  bounded/partial outcomes, explicit rejections, and top-level interface
  behavior. No other subagents were created.
- Stage 0 prerequisite:
  - `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py ...`
    included in final combined run below.
- Stage 1 expected failing baseline:
  - Initial focused run failed with missing
    `kazusa_ai_chatbot.coding_agent.code_reading` and missing exported
    `answer_code_question`, as expected before implementation.
- Stage 2 implementation created:
  - `src/kazusa_ai_chatbot/coding_agent/supervisor.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/__init__.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/models.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/agent.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/planner.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/evidence.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/synthesizer.py`;
  - `src/kazusa_ai_chatbot/coding_agent/code_reading/prompts.py`;
  - direct-interface contracts and exports in `coding_agent/models.py` and
    `coding_agent/__init__.py`;
  - direct-interface and Phase 2 handoff ICD updates in `coding_agent/README.md`.
- Stage 3 target acceptance:
  - `tests/test_coding_agent_image_reading_acceptance.py` verifies the Chinese
    target question through `code_reading.run(...)` and through
    `answer_code_question(...)` with a question-only embedded GitHub URL,
    Phase 0 parsing, and a deterministic managed-checkout handoff.
  - A manual direct `code_reading.run(...)` smoke on the current checkout
    returned `status="succeeded"` and a Chinese answer naming
    `base64_data`, `user_multimedia_input`, `multimedia_descriptor_agent`,
    `VISION_DESCRIPTOR_LLM`, `image_observation`, and `<image>` projection with
    evidence from adapters, service, brain graph, descriptor node,
    `cognition_episode`, `utils`, and persistence code.
- Stage 4 static boundary checks:
  - `rg -n "coding_agent" src\kazusa_ai_chatbot\background_work
    src\kazusa_ai_chatbot\action_spec
    src\kazusa_ai_chatbot\cognition_chain_core
    src\kazusa_ai_chatbot\service.py` returned no matches.
  - `rg -n "git clone|parse_github_source|parse_repo_hint|repo_url|
    local_root_hint|local_path_hint|requested_ref|source_scope_hint"
    src\kazusa_ai_chatbot\coding_agent\code_reading` returned no matches.
  - `git diff --check` passed with only Git line-ending normalization warnings
    for existing modified files.
- Stage 5 live direct-answer smoke:
  - Added `tests/test_coding_agent_live_llm.py`.
  - `venv\Scripts\python -m pytest -m live_llm
    tests\test_coding_agent_live_llm.py -q -s -k image_reading_question`
    skipped because `KAZUSA_CODING_AGENT_LIVE_SOURCE_URL` and
    `KAZUSA_CODING_AGENT_LIVE_WORKSPACE_ROOT` were not configured.
  - Live LLM answer quality was not verified.
- Stage 6 review:
  - User explicitly prohibited creating any subagent other than the test
    subagent, so the independent-code-review gate was performed by the parent
    agent as a user-approved exception to the review-subagent instruction.
  - Findings surfaced and fixed during review:
    - shared model circular import from `coding_agent.models` into
      `code_fetching`;
    - `rg` subprocess searches receiving captured stdin and searching zero
      bytes unless `.` is passed explicitly;
    - broad image-reading terms crowding out source pipeline evidence;
    - missing deterministic guard coverage for traversal, `.git`, `.env`,
      secret-like files, binary files, answer caps, and README ICDs;
    - missing conditional live diagnostic test;
    - missing top-level target acceptance through Phase 0 embedded-URL handoff.
  - Residual risks:
    - live LLM quality is unverified because live config was unavailable;
    - the current checkout cannot be used as a top-level `local_root_hint`
      smoke in this sandbox because Git reports dubious ownership, but the
      deterministic temporary Git checkout test covers the same Phase 0 local
      handoff under normal ownership.
- Final deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py
    tests\test_coding_agent_reading.py tests\test_coding_agent_interface.py
    tests\test_coding_agent_image_reading_acceptance.py -q`
    passed: `54 passed`.
  - `venv\Scripts\python -m py_compile` passed for changed production and test
    Python files.
