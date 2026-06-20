# coding_agent phase 1 reading plan

## Summary

- Goal: Add the standalone `code_reading` subagent package and top-level
  direct answer interface on top of the completed Phase 0 `code_fetching`
  contract.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: additive standalone module only; no Kazusa service,
  L2d, background-work, result-ready, or placeholder-removal integration in
  Phase 1.
- Highest-risk areas: unbounded source prompts, hallucinated code
  architecture, leaking raw source or absolute paths, bypassing the Phase 0
  fetching ICD, and failing the target image-reading question.
- Acceptance criteria: direct tests can call `answer_code_question(...)`, use
  Phase 0 `CodeFetchingResult`, read bounded source evidence, and return a
  Chinese answer explaining the image-reading pipeline with source-file
  evidence.

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
- Create `src/kazusa_ai_chatbot/coding_agent/code_reading/` as a subagent
  package.
- Add `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md` as the
  upstream ICD for the reading subagent.
- Add `code_reading.run(request: CodeReadingRequest) -> CodeReadingResult`.
- Implement reading over Phase 0 `CodeRepositoryRef` and `CodeSourceScope`
  using `rg --files`, `rg -n --json`, and bounded file reads.
- Implement evidence row assembly and answer synthesis.
- Add deterministic tests for reading planner behavior, evidence assembly,
  source-scope handling, prompt budget caps, answer response shape, README ICD
  consistency, and the target image-reading answer.
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

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stage dependency | Phase 1 depends on Phase 0 fetching. | Reading should consume a stable source contract instead of resolving storage itself. |
| Subagent packaging | `code_reading` is a folder with README ICD. | It is an owned subagent boundary. |
| Top-level interface | Add `answer_code_question(...)` in Phase 1. | The direct public answer API is meaningful only once reading exists. |
| Evidence | Synthesize only from bounded evidence rows. | Prevents unbounded source prompt leakage. |
| Source scope | Reading honors Phase 0 `repository`, `directory`, or `file` scope. | File and tree URLs should focus reading without losing repo context. |
| Fetching handoff | Phase 0 validates scoped GitHub paths and returns public-safe local source labels. | Reading can trust `repository.local_root` for filesystem access while keeping public metadata sanitized. |
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
    repository: CodeRepositoryRef | None
    source_scope: CodeSourceScope | None
    evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]
```

Public response fields must contain repo-relative evidence paths only. They
must not include raw command output, raw source dumps, secrets, adapter IDs,
job IDs, leases, retries, or absolute paths outside diagnostics.

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
- `tests/test_coding_agent_live_llm.py`: optional live direct-answer smoke.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/README.md`: add Phase 1 answer ICD.
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
   evidence rows, source-scope behavior, and `answer_code_question`.
3. Parent records expected failures before production-code edits.
4. Production-code subagent implements reading package, README ICD, supervisor,
   and direct answer entrypoint.
5. Parent runs focused reading and interface tests.
6. Parent runs target image-reading acceptance.
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

- [ ] Stage 0 - Phase 0 prerequisite verified
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_fetching.py -q`
  - Evidence: record pass output or Phase 0 blocker.
  - Sign-off: `<agent/date>`.

- [ ] Stage 1 - reading contract tests established
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_reading.py tests/test_coding_agent_interface.py -q`
  - Expected before implementation: missing module or missing entrypoint.
  - Evidence: record failure output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 2 - code_reading and direct answer implemented
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_reading.py tests/test_coding_agent_interface.py -q`
  - Evidence: record changed files and passing output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 3 - real demand acceptance complete
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_coding_agent_image_reading_acceptance.py -q`
  - Evidence: record generated answer excerpt and evidence list.
  - Sign-off: `<agent/date>`.

- [ ] Stage 4 - static boundary checks complete
  - Verify:
    `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
    returns no matches.
  - Verify: `git diff --check`.
  - Evidence: record grep and whitespace output.
  - Sign-off: `<agent/date>`.

- [ ] Stage 5 - live direct-answer smoke checked
  - Verify:
    `venv\Scripts\python.exe -m pytest -m live_llm tests/test_coding_agent_live_llm.py -q -s -k image_reading_question`
  - Evidence: record response status, evidence count, and missing-fact notes.
  - Sign-off: `<agent/date>`.

- [ ] Stage 6 - independent code review complete
  - Verify: rerun affected focused tests after review fixes.
  - Evidence: record findings, fixes, residual risks, and approval status.
  - Sign-off: `<agent/date>`.

## Verification

### Static Greps

- `rg "coding_agent" src/kazusa_ai_chatbot/background_work src/kazusa_ai_chatbot/action_spec src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/service.py`
  - Expected: no matches.
- `rg "git clone|repo_url|local_root_hint" src/kazusa_ai_chatbot/coding_agent/code_reading tests/test_coding_agent_reading.py`
  - Expected: no matches outside README explanations; reading must consume
    Phase 0 output, not fetch.

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

If live LLM configuration is unavailable, record the blocker and leave Stage 5
unchecked until live configuration is available or the user separately
approves deterministic-only completion.

### Static Checks

```powershell
git diff --check
```

## Independent Plan Review

Before approval, review this plan against the architecture reference and Phase
0 fetching plan. The reviewer must confirm Phase 1 consumes the Phase 0 ICD,
does not reimplement fetching, adds a `code_reading` README ICD, and keeps all
Kazusa runtime integration deferred.

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
- `code_reading` consumes `CodeFetchingResult`, `CodeRepositoryRef`, and
  `CodeSourceScope` instead of parsing URLs or cloning repos.
- `answer_code_question(...)` passes through all public Phase 0 source fields:
  `source_url`, `repo_url`, `repo_hint`, `local_root_hint`,
  `local_path_hint`, `requested_ref`, and `source_scope_hint`.
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

## Execution Evidence

### 2026-06-20 draft

- User split coding-agent staging:
  - Phase 0: `code_fetching`;
  - Phase 1: `code_reading`.
- User required each subagent to have a README ICD.
- Phase 1 now depends on Phase 0 fetching instead of implementing fetching.
- Draft status: not approved for implementation.
