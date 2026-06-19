# coding_agent phase 1 fetching and reading plan

## Summary

- Goal: Create a standalone top-level `coding_agent` module that can fetch or
  resolve a repository, read code with bounded tools, and return an
  evidence-backed answer to codebase questions through its own public
  interface.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: additive standalone module only; no Kazusa core
  service, L2d, background-work router, worker registry, or result-ready
  integration in Phase 1.
- Highest-risk areas: filesystem access, local LLM overreach, unbounded source
  prompts, accidental mutation of the active checkout, vague public interface,
  and failing to prove the real image-reading question through direct
  `coding_agent` tests.
- Acceptance criteria: direct tests can call the `coding_agent` public
  interface, resolve the local or fetched target repo, inspect code with
  `rg`/bounded reads, and return a Chinese answer explaining the image-reading
  pipeline with source-file evidence.

## Context

The long-term product direction is a specialized coding agent that can later
replace the current code-related placeholder behind Kazusa's background-work
path. The current background-work implementation and L2d/action-spec wiring are
not the Phase 1 target.

The user clarified the Phase 1 boundary:

```text
For phase 1 the plan is NOT to be connected to Kazusa's core service for now.
This will shrink your boundary so you can solely focus on the core coding agent
design. The coding agent will still be a top level module under Kauza's src repo
but for now the coding agent is independent, and for phase 1, the test will
directly interface with coding agent interface. Noted you will need the coding
agent interface ICD as README.md as the part of phase 1 deliverable.
```

The target real-world question remains:

```text
[eamars/KazusaAIChatbot](https://github.com/eamars/KazusaAIChatbot) 项目是怎么实现读图的
```

Phase 1 must implement only the standalone read-only coding path:

```text
CodingAgentRequest
  -> coding_agent.answer_code_question
  -> coding supervisor
  -> code_fetching
  -> code_reading
  -> CodingAgentResponse
```

`code_writing`, patch application, `code_executing`, background-worker
registration, L2d selection, action-spec changes, and result-ready cognition
delivery remain future integration work.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing coding supervisor prompts,
  reading prompts, LLM call counts, context budgets, or agent responsibility
  boundaries.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests that contain CJK
  string literals, including the Chinese image-reading acceptance question.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running live LLM coding-agent checks and before
  writing inspection artifacts for those checks.

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
- Keep Phase 1 independent from Kazusa's live cognition and background-worker
  runtime. Do not modify L2d, action-spec materialization, background-work
  routing, background-work provider dispatch, result-ready cognition, service
  delivery, adapters, persistence, consolidation, or scheduler code.
- Keep the public `coding_agent` interface stable and documented in
  `src/kazusa_ai_chatbot/coding_agent/README.md`.
- Keep coding tools behind deterministic allowlists, path checks, timeouts,
  size caps, and output caps.
- Do not mutate the active project checkout during Phase 1. The matching local
  repo may be read. Managed clones may be created only under the configured
  coding-agent workspace.
- Do not add compatibility shims, aliases, or fallback paths for future
  background-work integration.
- Do not allow raw tool output, raw source dumps, absolute paths outside the
  resolved repo, secrets, or adapter/service identifiers into public response
  metadata.

## Must Do

- Create `src/kazusa_ai_chatbot/coding_agent/` as a top-level standalone
  module.
- Add `src/kazusa_ai_chatbot/coding_agent/README.md` as the interface ICD for
  Phase 1. The README must define public entrypoints, request/response shapes,
  tool limits, refusal cases, safety boundaries, and direct-test usage.
- Add a public module entrypoint:
  `answer_code_question(request: CodingAgentRequest) -> CodingAgentResponse`.
- Add the top-level coding supervisor loop with Phase 1 actions:
  `code_fetching`, `code_reading`, `finish`, and `fail`.
- Implement `code_fetching` for public GitHub repository URL extraction, local
  checkout matching, managed clone creation, and repository metadata output.
- Implement `code_reading` for repository question answering using `rg --files`,
  `rg -n --json`, and bounded file reads.
- Add a deterministic code-tool facade with path containment, forbidden-file
  checks, command allowlists, explicit numeric limits, timeouts, and output
  caps.
- Add deterministic tests that call the standalone `coding_agent` public
  interface directly for tool safety, fetching, reading, supervisor behavior,
  response shape, README ICD consistency, and the target image-reading answer.
- Add one live LLM smoke test for the direct `coding_agent` interface, marked
  `live_llm`, that runs one case at a time when live LLM configuration is
  available.

## Deferred

- Do not connect `coding_agent` to Kazusa's background-work worker registry.
- Do not remove or decommission `text_artifact` in Phase 1.
- Do not update L2d, action-spec affordance wording, background-work router
  prompts, provider dispatch, result-ready cognition, L3/dialog, service
  delivery, adapters, background-work persistence, consolidation, or scheduler
  code.
- Do not update README/HOWTO runtime capability claims for Phase 1, except for
  this plan and the new `coding_agent` module README.
- Do not implement `code_writing`.
- Do not implement patch proposal, patch apply, or real workspace mutation.
- Do not implement `code_executing`.
- Do not install packages from a coding job.
- Do not add arbitrary shell access.
- Do not add private GitHub, SSH clone, credential handling, or authenticated
  repository support.
- Do not add worker-internal `web_agent3` calls in Phase 1.
- Do not redesign RAG2, L2d, L3, dialog, consolidation, dispatcher, or
  background-work job persistence.
- Do not remove legacy `background_artifact` compatibility processing for
  already queued old rows.

## Cutover Policy

Overall strategy: additive standalone module. Phase 1 has no Kazusa runtime
cutover.

| Area | Policy | Instruction |
|---|---|---|
| `coding_agent` module | additive | Create the standalone module under `src/kazusa_ai_chatbot/coding_agent/` with a documented public interface. |
| Kazusa core service path | no-op | Do not wire the module into L2d, background work, result-ready cognition, adapters, service delivery, or persistence. |
| Existing `text_artifact` worker | no-op | Keep existing production behavior unchanged in Phase 1. Placeholder removal belongs to a later integration plan. |
| Tests | additive | Add direct interface tests for `coding_agent`; do not rewrite service/background-work tests for this phase. |
| Repository mutation | bigbang no-op | Phase 1 forbids mutation entirely. |
| Future integration | deferred | Later plans may connect the public interface to a background worker and remove placeholders. This plan does not authorize that work. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not expand the plan into a service integration by default.
- For no-op areas, leave existing runtime behavior unchanged.
- For additive areas, introduce only the listed module, README ICD, tests, and
  direct interface.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

After Phase 1, developers can call `kazusa_ai_chatbot.coding_agent` directly
from tests or a future integration layer. The module resolves a repository,
reads relevant source files through bounded tools, synthesizes an
evidence-backed answer, and returns a typed response without touching Kazusa's
live service path.

The new module README is the Phase 1 ICD. A future background worker can depend
on the README-defined public interface without importing internals.

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

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Runtime placement | Implement Phase 1 as a standalone top-level module under `src/kazusa_ai_chatbot/coding_agent/`. | The user explicitly narrowed Phase 1 away from Kazusa core-service integration. |
| Public contract | Treat `src/kazusa_ai_chatbot/coding_agent/README.md` as the interface ICD. | Future background-worker integration needs a stable boundary that avoids importing internals. |
| Test entrypoint | Direct tests call `answer_code_question(...)`. | This proves the core coding-agent design without depending on L2d, queues, workers, or dialog. |
| Tooling | Use `git` and `rg` through deterministic subprocess wrappers. | Avoids reinventing mature code discovery/search tools while preserving local safety checks. |
| Repository scope | Prefer existing matching local checkout; otherwise clone public HTTPS GitHub repos into a managed workspace. | Supports the target repo while keeping active checkout mutation forbidden. |
| Answer synthesis | Let `code_reading` synthesize from bounded evidence rows. | The LLM sees evidence, not unbounded source files or raw command output. |
| LLM route | Use the existing background-work LLM configuration through local module ownership for Phase 1. | Avoids adding route configuration before Kazusa runtime integration exists. |
| External help | Defer `web_agent3` calls to a later plan. | The target question is answerable from source code; adding web help now increases blast radius. |

## Contracts And Data Shapes

### Public Entrypoint

`src/kazusa_ai_chatbot/coding_agent/__init__.py` must expose only the stable
Phase 1 interface:

```python
async def answer_code_question(
    request: CodingAgentRequest,
) -> CodingAgentResponse:
    ...
```

Future integration layers must call this entrypoint instead of importing
`code_fetching`, `code_reading`, `tools`, `supervisor`, or `prompts` directly.

### Request

```python
class CodingAgentRequest(TypedDict, total=False):
    question: str
    repo_url: str
    repo_hint: str
    local_root_hint: str
    workspace_root: str
    preferred_language: str
    max_answer_chars: int
```

Required effective inputs:

- `question` is required and must be non-empty after stripping.
- At least one of `repo_url`, `repo_hint`, or `local_root_hint` is required.
- `preferred_language` defaults to `"zh"` for the target acceptance case.
- `max_answer_chars` defaults to `3000` and is bounded between `500` and
  `8000`.

### Response

```python
class CodingAgentResponse(TypedDict):
    status: Literal["succeeded", "failed", "needs_user_input", "rejected"]
    answer_text: str
    repository: CodeRepositoryRef | None
    evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]
```

Public response fields must contain repo-relative paths only. They must not
include raw command output, absolute paths outside the resolved repo, secrets,
adapter identifiers, job identifiers, leases, retries, or background-worker
metadata.

### Supervisor Decision

```python
class CodingSupervisorDecision(TypedDict):
    action: Literal["code_fetching", "code_reading", "finish", "fail"]
    instruction: str
    reason: str
```

The normal Phase 1 transition sequence is:

```text
start -> code_fetching -> code_reading -> finish
```

The supervisor may perform one additional `code_reading` pass when the first
answer lacks source evidence. `CODING_AGENT_SUPERVISOR_MAX_CYCLES` must default
to `4` and be bounded between `1` and `6`.

### Repository Result

```python
class CodeRepositoryRef(TypedDict):
    source_url: str
    provider: Literal["github"]
    owner: str
    repo: str
    local_root: str
    current_commit: str
    default_branch: str
    managed_checkout: bool
    dirty_state: Literal["clean", "dirty", "unknown"]
```

`local_root` is internal state. Public responses and evidence must use
repo-relative paths, not `local_root`, unless an error message needs to name the
user-provided `local_root_hint`.

### Tool Result

```python
class CodeToolResult(TypedDict):
    status: Literal["succeeded", "failed", "rejected", "not_found"]
    command_kind: str
    stdout: str
    stderr: str
    exit_code: int | None
    truncated: bool
    metadata: dict[str, object]
```

Tool results passed to LLM prompts must be compacted into evidence rows before
prompting.

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

`path` must be repo-relative. `excerpt` must be capped before entering prompts
or public responses.

### Tool Limits

Implement these numeric defaults in one visible constants block or equivalent
configuration object inside `coding_agent`:

| Limit | Default |
|---|---:|
| prompt budget | 50,000 characters |
| final answer default | 3,000 characters |
| final answer maximum | 8,000 characters |
| file read maximum | 20,000 bytes per file |
| evidence excerpt maximum | 1,200 characters per row |
| search match maximum | 80 matches per search |
| search line maximum | 240 characters |
| evidence row maximum | 12 rows |
| `rg` subprocess timeout | 30 seconds |
| `git` metadata timeout | 30 seconds |
| `git clone` timeout | 120 seconds |
| listed source file maximum | 2,000 files |

The tool facade must reject `.env`, secret-like files, `.git` internals,
binary files, paths outside the repo root, and unsupported command kinds.

## LLM Call And Context Budget

Before this plan:

- `coding_agent` does not exist.
- No direct coding-agent LLM calls exist.

After this plan:

- Kazusa live path: no new LLM calls.
- Kazusa background-worker path: no new LLM calls.
- Direct `coding_agent` path:
  - coding supervisor: up to four background-work-route LLM calls;
  - code reading planner/synthesizer: up to two background-work-route LLM
    calls;
  - no `web_agent3` calls in Phase 1.

Context budget:

- Use `50k characters` as the maximum coding-agent prompt budget unless a
  lower model limit is configured.
- Each file excerpt is capped before entering prompts.
- Search results are capped by match count and line length.
- Full source files are never inserted into prompts by default.
- The final `answer_text` is capped by `max_answer_chars`.

Latency impact:

- Phase 1 has no user-facing live-path latency impact because the module is not
  wired into the service.
- Direct tests and future callers must treat the interface as potentially slow
  because it may clone repositories, run `rg`, and make multiple LLM calls.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/README.md`: Phase 1 interface ICD,
  public entrypoint contract, request/response shapes, tool limits, refusal
  cases, safety rules, direct-call example, and future-integration note.
- `src/kazusa_ai_chatbot/coding_agent/__init__.py`: public entrypoint exposing
  `answer_code_question` and public TypedDict contracts.
- `src/kazusa_ai_chatbot/coding_agent/models.py`: Phase 1 TypedDict contracts
  and numeric limit constants.
- `src/kazusa_ai_chatbot/coding_agent/tools.py`: deterministic `git`, `rg`,
  and bounded file-read facade.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching.py`: repository URL
  extraction, local checkout matching, managed clone resolution.
- `src/kazusa_ai_chatbot/coding_agent/code_reading.py`: reading plan, search
  execution, evidence assembly, answer synthesis.
- `src/kazusa_ai_chatbot/coding_agent/supervisor.py`: top-level bounded
  supervisor loop and public-entrypoint orchestration.
- `src/kazusa_ai_chatbot/coding_agent/prompts.py`: static prompt constants for
  supervisor and reading.
- `tests/test_coding_agent_tools.py`: path safety, forbidden files, command
  allowlist, output caps, and numeric limit behavior.
- `tests/test_coding_agent_fetching.py`: GitHub URL extraction, local checkout
  matching, managed clone command construction, and refusal cases.
- `tests/test_coding_agent_reading.py`: reading evidence assembly and image
  pipeline answer synthesis with patched tool output.
- `tests/test_coding_agent_interface.py`: public entrypoint, request
  validation, response shape, and README ICD consistency.
- `tests/test_coding_agent_image_reading_acceptance.py`: deterministic direct
  interface answer for the target image-reading question with local repo
  evidence.
- `tests/test_coding_agent_live_llm.py`: one-case live LLM smoke test for the
  direct public interface, marked `live_llm`.

### Modify

- `development_plans/active/short_term/coding_agent_phase1_fetching_reading_plan.md`:
  keep the active plan aligned with the standalone Phase 1 boundary.
- `development_plans/README.md`: keep the registry row description aligned with
  the standalone Phase 1 boundary.

### Keep

- `src/kazusa_ai_chatbot/background_work/*`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/action_spec/*`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/cognition_chain_core/*`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/service.py`: no Phase 1 changes.
- `src/kazusa_ai_chatbot/background_artifact/*`: keep legacy compatibility for
  old rows only.
- `src/kazusa_ai_chatbot/rag/web_agent3/*`: no Phase 1 changes.
- Top-level `README.md` and `docs/HOWTO.md`: no Phase 1 runtime capability
  claim changes.

## Overdesign Guardrail

- Actual problem: Kazusa's repo does not yet have a standalone coding-agent
  core that can fetch/read repositories and answer codebase questions through a
  stable interface.
- Minimal change: add one read-only top-level `coding_agent` module with an
  ICD README, a direct public interface, `code_fetching`, `code_reading`, a
  supervisor, and deterministic `git`/`rg` tools.
- Ownership boundaries: the public interface owns request/response shape;
  coding supervisor owns coding subagent order; `code_fetching` owns repository
  resolution; `code_reading` owns evidence assembly and answer synthesis;
  deterministic tools own filesystem safety and command execution.
- Rejected complexity: service integration, background-worker registration,
  L2d routing, result-ready cognition, placeholder removal, patch proposal,
  patch apply, arbitrary shell, package installation, Docker execution, private
  repositories, authenticated GitHub, direct adapter send, direct cognition
  calls, RAG2 redesign, web-agent calls, and compatibility fallbacks.
- Evidence threshold: add deferred integration only after Phase 1 passes the
  direct image-reading acceptance case and a later approved plan names the
  integration target.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan and the module README ICD.
- The responsible agent must not introduce new architecture, alternate
  integration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside
  `src/kazusa_ai_chatbot/coding_agent/`, listed tests, this plan, and the plan
  registry row as out of scope.
- The responsible agent must search for existing equivalent helpers before
  adding a new helper.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the
  plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent establishes the focused direct-interface test contract before
   production-code edits.
2. Parent runs focused tests and records expected failures or current baseline.
3. Parent starts one production-code subagent with this plan, mandatory skills,
   the README ICD requirement, and the focused tests.
4. Production-code subagent creates the standalone coding-agent package and
   README ICD.
5. Parent adds or updates acceptance and live-LLM tests while the
   production-code subagent works.
6. Parent runs focused direct-interface tests.
7. Parent runs deterministic acceptance tests for the target image-reading
   question.
8. Parent runs static checks and the non-live deterministic suite.
9. Parent runs live LLM checks one case at a time when live LLM configuration
   is available.
10. Parent starts one independent code-review subagent.
11. Parent remediates review findings inside the approved change surface and
   reruns affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue acceptance tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused interface test contract established
  - Covers: public entrypoint, README ICD consistency, request validation,
    response shape, tool safety, fetching, reading, and the target
    image-reading answer.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_interface.py tests/test_coding_agent_tools.py tests/test_coding_agent_fetching.py tests/test_coding_agent_reading.py -q`
  - Expected before implementation: fails with missing module, missing
    entrypoint, or missing README ICD.
  - Evidence: record failure output in `Execution Evidence`.
  - Handoff: next agent starts Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - coding-agent production package implemented
  - Covers: `coding_agent` package, deterministic tools, `code_fetching`,
    `code_reading`, supervisor, public entrypoint, and README ICD.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_interface.py tests/test_coding_agent_tools.py tests/test_coding_agent_fetching.py tests/test_coding_agent_reading.py -q`
  - Evidence: record changed files and passing output.
  - Handoff: next agent starts Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - real demand acceptance path complete
  - Covers: deterministic direct-interface answer for the image-reading
    question with local repo evidence.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_image_reading_acceptance.py -q`
  - Evidence: record the generated answer excerpt and file evidence list.
  - Handoff: next agent starts Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - static and regression verification complete
  - Covers: no accidental service integration, no forbidden placeholder
    removal, line/whitespace checks, and non-live deterministic suite.
  - Verify:
    `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
    returns no matches.
  - Verify:
    `git diff --check`
  - Verify:
    `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`
  - Evidence: record static grep output and test output.
  - Handoff: next agent starts Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - live LLM smoke checked one case at a time
  - Covers: direct `coding_agent` public interface answer generation with
    configured live LLM.
  - Verify:
    `venv\Scripts\python.exe -m pytest -m live_llm tests/test_coding_agent_live_llm.py -q -s -k image_reading_question`
  - Evidence: record trace path, response status, evidence count, and whether
    the generated answer contains required image-pipeline facts.
  - Handoff: next agent starts Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - independent code review complete
  - Covers: full implementation diff, plan alignment, README ICD accuracy,
    verification evidence, prompt safety, path safety, standalone boundary, and
    no accidental Kazusa service integration.
  - Verify: run affected focused tests again after review fixes.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and approval status.
  - Handoff: plan may be marked completed only after this gate passes.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
  - Expected after implementation: no matches. A nonzero `rg` exit because no
    matches are found is acceptable.
- `rg "background_work|background_work_result_ready|L2d|action_spec|text_artifact" src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_*.py`
  - Expected after implementation: matches only in README future-integration
    notes or tests that assert Phase 1 does not integrate with those systems.
- `rg "answer_code_question|CodingAgentRequest|CodingAgentResponse" src/kazusa_ai_chatbot/coding_agent tests/test_coding_agent_*.py`
  - Expected after implementation: matches the public interface, README ICD,
    and direct tests.

### Focused Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_coding_agent_interface.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_tools.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_reading.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_image_reading_acceptance.py -q
```

### Live LLM Smoke

Run one case at a time:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests/test_coding_agent_live_llm.py -q -s -k image_reading_question
```

If live LLM configuration is unavailable, do not mark Stage 5 complete. Record
the blocker and leave the plan unfinished or ask the user whether deterministic
verification is sufficient for the current environment.

### Static Checks

```powershell
git diff --check
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

The broader deterministic suite may be split into batches if runtime is high,
but every failing batch must be inspected before sign-off.

## Independent Plan Review

Before approving this plan for implementation, run an independent plan review
with a fresh agent. The reviewer must check:

- the architecture aligns with
  `development_plans/reference/designs/coding_agent_architecture.md` while
  respecting the narrowed standalone Phase 1 boundary;
- the Phase 1 scope is limited to a top-level independent module, direct public
  interface, README ICD, fetching, reading, supervisor, deterministic tools,
  and direct tests;
- no Kazusa core-service, L2d, action-spec, background-work, result-ready,
  adapter, persistence, consolidation, scheduler, or placeholder-removal work
  remains in Phase 1 scope;
- tool access is deterministic, bounded, and path-safe;
- no prompt asks a local LLM to infer filesystem safety, permissions, command
  validity, adapter delivery, persistence, or integration routing;
- the target image-reading question has a deterministic direct-interface
  acceptance path;
- live LLM verification is one case at a time;
- the README ICD is specific enough for a future integration layer to call the
  public interface without importing internals.

Record blockers, non-blocking findings, required edits, and approval status in
`Execution Evidence` before moving this plan out of `draft`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, filesystem
  safety, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- README ICD accuracy, public entrypoint stability, direct-test coverage, and
  absence of accidental Kazusa service integration.
- Regression and handoff quality, including focused and regression tests,
  execution evidence, next-stage handoff notes, and path-safe commands for
  Windows paths.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `src/kazusa_ai_chatbot/coding_agent/` exists as a standalone top-level
  module.
- `src/kazusa_ai_chatbot/coding_agent/README.md` defines the Phase 1 interface
  ICD, including public entrypoint, request/response shapes, tool limits,
  refusal cases, safety rules, and direct-test usage.
- `answer_code_question(request: CodingAgentRequest) -> CodingAgentResponse`
  is the only public entrypoint that future integrations need.
- Direct deterministic tests call the public interface without using L2d,
  action-specs, background-work jobs, worker registry, result-ready cognition,
  service delivery, adapters, or persistence.
- `code_fetching` can resolve the target GitHub repo to the local checkout or
  a managed public clone.
- `code_reading` can answer the image-reading question in Chinese with source
  evidence.
- No Phase 1 code path writes files in the active checkout, applies patches,
  installs packages, runs arbitrary shell commands, mutates the active
  checkout, or sends adapter text directly.
- No Phase 1 diff modifies Kazusa core-service integration files listed as
  `Keep`.
- Focused tests, static greps, `git diff --check`, and the non-live
  deterministic test suite pass or have documented user-approved exceptions.
- The live LLM smoke case passes when live LLM configuration is available, or
  the plan remains unfinished with the live LLM blocker recorded.
- Independent code review passes with findings resolved or explicitly recorded
  as residual risks accepted by the user.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Filesystem access leaks secrets | Refuse `.env`, secret-like files, `.git` internals, binary files, and paths outside repo root. | `tests/test_coding_agent_tools.py` and static safety review. |
| Public interface is too vague for later integration | Make README ICD a deliverable and test it against exported symbols and request/response shapes. | `tests/test_coding_agent_interface.py` and independent code review. |
| Local LLM hallucinates architecture | Answer synthesis receives only bounded evidence rows and must list file evidence. | Image-reading acceptance test checks required evidence topics. |
| Active checkout is mutated | Phase 1 forbids pull/checkout/apply for active checkout. | Fetching tests and code review. |
| Phase 1 accidentally expands into Kazusa service integration | Mark service/background/L2d/action-spec files as `Keep` and grep for accidental references. | Stage 4 static greps and diff review. |
| Live direct-interface LLM answer is weak | Add one live case and inspect evidence/facts, but keep deterministic acceptance as the primary Phase 1 gate. | Stage 5 live LLM smoke. |
| Raw tool output leaks into public response | Compact tool output into evidence rows and cap excerpts before prompts/responses. | Interface, reading, and tool tests plus code review. |

## Execution Evidence

### 2026-06-19 draft

- User approved creating two planning documents:
  - coding-agent architecture reference;
  - Phase 1 `code_fetching`/`code_reading`/supervisor development plan.
- Repo planning rules were inspected:
  - `development_plans/README.md`;
  - `.agents/skills/development-plan/SKILL.md`;
  - `plan_contract.md`;
  - `cutover_policy.md`;
  - `execution_gates.md`.
- Current architecture context was inspected:
  - `README.md`;
  - `docs/HOWTO.md`;
  - `src/kazusa_ai_chatbot/background_work/README.md`;
  - `src/kazusa_ai_chatbot/action_spec/README.md`;
  - `src/kazusa_ai_chatbot/nodes/README.md`;
  - `src/kazusa_ai_chatbot/brain_service/README.md`;
  - background-work router, providers, worker, jobs, delivery, result-source,
    and current `text_artifact` worker;
  - action-spec background-work handler and registry;
  - `web_agent3` public contract and README.
- Draft status: not approved for implementation.

### 2026-06-19 scope clarification

- User clarified that Phase 1 is not connected to Kazusa's core service.
- Phase 1 boundary changed to a standalone top-level
  `src/kazusa_ai_chatbot/coding_agent/` module with direct interface tests.
- `src/kazusa_ai_chatbot/coding_agent/README.md` is now a required Phase 1 ICD
  deliverable.
- Background-work, L2d, action-spec, result-ready cognition, service delivery,
  and placeholder removal moved to deferred integration work.
