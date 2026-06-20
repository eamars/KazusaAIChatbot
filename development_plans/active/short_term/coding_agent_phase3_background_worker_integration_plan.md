# coding agent phase3 background worker integration plan

## Summary

- Goal: integrate completed standalone `coding_agent` capabilities as a Kazusa
  `background_work` worker named `coding_agent`, so supported code tasks can
  enter through L2d, run after the live turn, and return through
  `background_work_result_ready` cognition.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `debug-llm`, `superpowers:test-driven-development`,
  `superpowers:subagent-driven-development`, and
  `superpowers:requesting-code-review`.
- Overall cutover strategy: compatible for the existing generic
  `background_work` queue and text-artifact worker; bigbang for code-repository
  task ownership, because repository/codebase reading must route to
  `coding_agent` with no fallback to the text-artifact placeholder.
- Highest-risk areas: integrating before standalone phases are stable, leaking
  worker names or local paths into L2d/L3 prompts, losing the original user
  code task when the worker receives only route reason context, overloading the
  text-artifact worker with repository work, and adding deterministic keyword
  routing over user text.
- Acceptance criteria: completed standalone capabilities selected for Phase 3
  are accepted; `coding_agent` is discoverable as a background-work worker;
  supported code tasks route to that worker; worker results are sanitized and
  delivered through existing result-ready cognition; text-artifact behavior
  still works; all deterministic, static, and one-case live LLM gates in this
  plan pass.

## Context

Phase 0 code fetching is archived as completed. Phase 1 is the required
standalone code-reading contract on top of Phase 0. Phase 2 is standalone
`code_writing`. This Phase 3 plan must not execute until the selected
standalone capabilities from Phases 1 and 2 are completed and accepted with
their public direct interfaces, including Phase 1 `answer_code_question(...)`.

Phase 2 will define the standalone `code_writing` public interface and patch
proposal response contract. This Phase 3 draft must consume that completed
contract as an adapter boundary; it must not invent the writing interface.

This draft was originally written before Phase 2 was corrected to
`code_writing`. Before Phase 3 approval or execution, reread the completed
Phase 2 plan and update this draft's worker mapping, tests, and acceptance
criteria to match the actual Phase 2 public contract.

Phase 3 is the first Kazusa runtime integration stage. The current
`background_work` runtime has only one production worker,
`subagent.text_artifact`. That worker is intentionally text-only. It can create
bounded code snippets, rewrites, and summaries, but it must not fetch
repositories, inspect files, use `rg`, answer source-code architecture
questions, write patches, run commands, install packages, or send adapter text.

The existing queue, job document, worker loop, result-ready cognition delivery,
and action-spec boundary are the correct runtime foundation. L2d already
selects only a generic private `background_work_request`; deterministic code
materializes `task_brief`, delivery scope, and queue rows; the background-work
router selects a worker after the live turn; completed jobs re-enter cognition
as `background_work_result_ready`.

Two current implementation details must be corrected for Phase 3:

- `background_work.router.normalize_background_work_router_output(...)`
  currently accepts only `text_artifact`; Phase 3 must allow enabled worker
  names from the worker registry, including `coding_agent`.
- `background_work.worker.run_background_work_worker_tick(...)` currently
  builds a worker-facing `source_summary` from `source_context` when present,
  otherwise from `task_brief`. For code questions, `task_brief` is the trusted
  user task and may contain the repository URL or raw-file URL. The route
  reason in `source_context` must not replace it before worker dispatch.

## Mandatory Skills

- `development-plan`: load before editing this plan, approving it, executing
  it, updating lifecycle records, or reviewing completion.
- `local-llm-architecture`: load before changing background-work routing,
  prompts, coding-agent handoff, LLM configuration, or context budgets.
- `no-prepost-user-input`: load before changing L2d/action-selection,
  background-work routing, or code that could classify user requests by local
  keywords.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running live/local LLM diagnostics or writing
  debug-LLM-style reports.
- `superpowers:test-driven-development`: load before implementation; parent
  must establish focused failing tests before production-code changes.
- `superpowers:subagent-driven-development`: load before execution; normal
  execution uses one production-code subagent and one independent code-review
  subagent.
- `superpowers:requesting-code-review`: load before the final independent
  review gate.

## Mandatory Rules

- After context compaction or any major checklist sign-off, the parent or
  active execution agent must reread this entire plan before continuing.
- Before approval or execution, run `Independent Plan Review` and resolve all
  blockers.
- Before final completion, lifecycle updates, merge, or sign-off, run
  `Independent Code Review` and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Execution is blocked until Phase 1 and Phase 2 selected standalone
  capabilities are completed and accepted. If
  `development_plans/active/short_term/coding_agent_phase1_code_reading_final_plan.md`
  remains `draft`, `approved`, or `in_progress`, stop before production-code
  changes. If the Phase 2 code-writing plan is not completed, stop before
  production-code changes.
- Phase 3 must consume only public standalone coding-agent interfaces and
  public response fields. Do not import PM, programmer, planner,
  repository-map, synthesizer, or tool internals from the background worker.
- L2d must continue to see only generic `background_work_request`. Do not
  expose worker names, repository paths, tool args, source URLs as separate
  fields, patch content, shell commands, job ids, leases, adapter ids, or final
  answer text to L2d.
- Deterministic code must not keyword-route user text into `coding_agent`.
  L2d semantically decides whether background work is appropriate; the
  background-work router semantically chooses the enabled worker. Python owns
  structural validation only.
- Deterministic action materialization remains the owner of `task_brief`.
  `task_brief` is trusted worker input and must not be replaced by
  `source_context` or router reason before worker execution.
- The coding worker must pass `CODING_AGENT_WORKSPACE_ROOT` from configuration
  into the public standalone coding-agent request. It must not parse workspace
  paths from user text, use Phase 0's temp fallback, or invent a process-local
  workspace.
- If `CODING_AGENT_WORKSPACE_ROOT` is absent or empty, the coding worker must
  return a sanitized non-success `BackgroundWorkResult` and must not fetch or
  read code.
- Phase 3 is integration-only. Do not implement code writing, patch proposal,
  patch apply, command execution, package installation, Docker execution,
  broad web help, or external documentation lookup as part of this plan.
- Workers must never call adapters, dispatcher delivery, cognition, or service
  graph entrypoints directly. Workers return only `BackgroundWorkResult`; the
  existing delivery layer builds result-ready cognition.
- `artifact_text`, summaries, and `worker_metadata` must not expose absolute
  local paths, `workspace_root`, `local_root`, cache keys, raw command output,
  raw source excerpts in metadata, `.env` content, `.git` internals, job ids,
  leases, adapter ids, or delivery fields.
- Phase 3 must reuse the standalone coding-agent effective LLM route decision:
  optional `CODING_AGENT_LLM_*` with fallback to `BACKGROUND_WORK_LLM_*`. Do
  not add `CODING_AGENT_WORKER_LLM`, `CODING_AGENT_PM_LLM`,
  `CODING_AGENT_PROGRAMMER_LLM`, or `CODING_AGENT_SYNTHESIZER_LLM`.

## Must Do

- Verify that Phase 1 and Phase 2 selected standalone capabilities are
  completed and that their focused direct tests pass before starting Phase 3
  production-code changes.
- Add `CODING_AGENT_WORKSPACE_ROOT` configuration and documentation; it is
  optional at startup and required for executing a `coding_agent` job.
- Add and register `kazusa_ai_chatbot.background_work.subagent.coding_agent`
  as the worker adapter over public standalone coding-agent interfaces, with a
  prompt-safe description for supported code tasks only.
- Update background-work router prompt and normalization so the router can
  select any enabled worker from the registry and still emits only `action`,
  `worker`, and `reason`.
- Preserve route-only L2d/action-spec boundaries while broadening the
  background-work affordance beyond text-only work.
- Fix dispatch so worker execution receives trusted `task_brief` separately
  from route/source summary context.
- Map standalone coding-agent responses into `BackgroundWorkResult` exactly as
  specified by the completed standalone ICDs.
- Keep text-artifact snippets/rewrites/summaries, but remove repository or
  codebase reading ownership from text-artifact wording and tests.
- Add deterministic tests for worker registration, router selection,
  task-brief handoff, response mapping, missing workspace, sanitization, and
  result-ready delivery projection.
- Add one frozen L2d fixture, one L2d live LLM case, one router live LLM case,
  and one Phase 3 worker live/local diagnostic for a completed standalone
  reading or writing capability selected for this integration pass.
- Update the root README, HOWTO, background-work ICD, and coding-agent ICD.

## Deferred

- Do not implement `code_writing`, patch proposal logic, patch apply, or
  `code_executing`; Phase 2 owns standalone writing.
- Do not add a direct synchronous `/chat` code-answer path.
- Do not let L2d choose `coding_agent` or any worker-local task type.
- Do not add a second durable job collection or a coding-agent-specific
  delivery pipeline.
- Do not expose `worker_metadata` to normal prompt payloads unless a later
  approved plan changes the result-ready cognition contract.
- Do not add repository auto-pull behavior beyond whatever Phase 0 completed.
- Do not add broad `web_agent3` or external-help calls in Phase 3.
- Do not migrate legacy `background_artifact` rows or delete the
  `background_artifact` compatibility modules.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Existing generic background-work queue | compatible | Keep `background_work_jobs`, queue validation, worker loop, delivery, and result-ready cognition schema. Add the coding worker without creating a new queue. |
| Text-artifact worker | compatible | Keep snippets, rewrites, and summaries. Tighten wording so repository/codebase reading belongs to `coding_agent`, not text-artifact. |
| Code-related background ownership | bigbang | Route repository/source-code reading jobs to `coding_agent`; do not fall back to text-artifact for repository reading. |
| Router worker enum | bigbang | Replace hardcoded `text_artifact` normalization with enabled-worker validation from the registry. No unknown-worker fallback. |
| L2d/action-spec contract | compatible | Preserve `background_work_request` as route-only. Update semantic affordance wording only. |
| Worker decision context | bigbang | Pass trusted `task_brief` to workers; do not overload `source_summary` with route reason as the only worker input. |
| LLM routes | compatible | Reuse Phase 1 effective `CODING_AGENT_LLM` fallback to `BACKGROUND_WORK_LLM`; add no new coding model route. |

## Target State

A user can ask a codebase reading question through the normal Kazusa chat path,
for example:

```text
[eamars/KazusaAIChatbot](https://github.com/eamars/KazusaAIChatbot) project:
how is image reading implemented?
```

The live turn remains bounded:

```text
user message
-> L2d selects speak + background_work_request
-> deterministic action materialization builds task_brief and queue scope
-> durable background_work_jobs row is queued
-> L3/dialog acknowledges that background work was accepted
```

The slow work runs outside the live turn:

```text
background_work worker tick
-> background_work router selects worker="coding_agent"
-> coding_agent background worker builds the standalone request
-> public standalone coding-agent interface
-> Phase 0 fetching as needed
-> Phase 1 reading and/or Phase 2 writing as selected by completed ICDs
-> standalone coding-agent response artifact
-> BackgroundWorkResult
-> durable job completed or failed
-> background_work_result_ready cognition
-> L3/dialog renders visible result through existing delivery
```

The result-ready cognition receives only bounded artifact text, summaries, and
sanitized metadata. It does not receive raw checkout roots, workspace roots,
cache keys, raw source excerpts in metadata, raw tool output, job leases, or
adapter delivery fields.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Runtime placement | Implement `coding_agent` as a `background_work` worker | Coding tasks are slow and tool-using, so they must run after the live persona turn. |
| Worker interface | Adapt only public standalone coding-agent interfaces | Prevents the background worker from depending on PM/programmer internals and keeps prior phases as the core contract. |
| L2d exposure | Keep a single generic `background_work_request` | L2d decides semantic need for background work, not worker choice or code-tool parameters. |
| Router choice | Background-work router selects `coding_agent` from registry descriptions | Preserves the existing router/subagent split and avoids deterministic keyword routing. |
| Worker task text | Pass trusted `task_brief` to worker execution | Code questions need the original decontextualized task and URL; route reason is not a valid substitute. |
| Workspace root | Use `CODING_AGENT_WORKSPACE_ROOT` as worker configuration | Phase 3 cannot rely on direct-call temp fallback or user-supplied filesystem paths. |
| Result contract | Reuse `BackgroundWorkResult` | Existing job persistence and delivery already own background result lifecycle. |
| Metadata | Store sanitized repository/source/evidence references in `worker_metadata`, but do not project raw metadata into normal prompt payloads | Keeps operator/debug value without widening L3 prompt exposure. |
| Placeholder removal | Keep text-artifact for text artifacts, remove repository/code work from its ownership | Prevents coding work from being handled by the placeholder while preserving existing text jobs. |
| Writing handoff | Add no new queue or L2d action for writing in Phase 3 | Completed Phase 2 writing can be integrated behind the same coding-worker boundary after this draft is revised against the Phase 2 contract. |

## Contracts And Data Shapes

### Configuration

Phase 3 adds this environment setting:

```text
CODING_AGENT_WORKSPACE_ROOT=<absolute or repo-relative directory path>
```

Rules:

- Empty or absent keeps `coding_agent` registered but unavailable for execution.
- Service startup must not fail solely because this value is absent.
- A claimed `coding_agent` job with an empty value returns `status="failed"` or
  `status="rejected"` with a sanitized configuration failure summary.
- The value is passed into standalone coding-agent requests as
  `workspace_root`.
- The value must never appear in `BackgroundWorkResult`, job prompt payloads,
  event logs intended for model context, or visible dialog.

### Router Decision

`BackgroundWorkRouterDecision` remains route-only:

```python
{
    "action": "execute | reject | needs_user_input | stop",
    "worker": "enabled worker name | none",
    "reason": "short route reason",
}
```

The router normalizer must validate `worker` against the enabled worker names
provided to the router. Unknown workers normalize to `"none"`, and non-execute
actions force `"worker": "none"`.

The router output must not contain:

```text
task_brief, source_url, repo_url, local_root_hint, workspace_root, work_kind,
task_type, tool_args, artifact_text, job_id, adapter_id, delivery fields,
patches, shell commands
```

### Worker Decision Context

`BackgroundWorkWorkerDecision` gains deterministic worker context while keeping
router output unchanged:

```python
{
    "action": "execute",
    "worker": "coding_agent",
    "reason": str,
    "task_brief": str,
    "source_summary": str,
}
```

Rules:

- `task_brief` is copied from the durable job row and is the primary task text
  for the worker.
- `source_summary` is optional contextual routing/source summary. It must not
  replace `task_brief`.
- `task_brief` is deterministic worker execution context, not an LLM router
  output field and not an L2d prompt field.

### Coding Worker Entrypoint

Create:

```python
WORKER = "coding_agent"
DESCRIPTION = (
    "Answers read-only source-code and repository questions from bounded "
    "local source evidence. Does not write patches, execute commands, "
    "install packages, or create generic text artifacts."
)

async def execute(
    decision: BackgroundWorkWorkerDecision,
    *,
    max_output_chars: int,
) -> BackgroundWorkResult:
    ...
```

The worker builds:

```python
{
    "question": decision["task_brief"],
    "workspace_root": CODING_AGENT_WORKSPACE_ROOT,
    "max_answer_chars": max_output_chars,
}
```

The worker does not pass `local_root_hint`, `local_path_hint`, `repo_hint`, or
`source_url` unless those fields exist in a future trusted background job
contract. In Phase 3, Phase 0 extracts supported source URLs from `question`.

### Result Mapping

Map `CodingAgentResponse` to `BackgroundWorkResult` exactly:

```python
{
    "status": response["status"],
    "worker": "coding_agent",
    "artifact_text": response["answer_text"][:max_output_chars]
        if response["status"] == "succeeded"
        else "",
    "failure_summary": failure_summary,
    "result_summary": result_summary,
    "worker_metadata": {
        "schema_version": "coding_agent_worker_metadata.v1",
        "coding_operation": "code_reading",
        "repository": response["repository"],
        "source_scope": response["source_scope"],
        "evidence_refs": [
            {
                "path": row["path"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "symbol_or_topic": row["symbol_or_topic"],
                "reason": row["reason"],
            }
            for row in response["evidence"]
        ],
        "limitations": response["limitations"],
        "trace_summary": response["trace_summary"],
    },
}
```

`failure_summary` rules:

- `""` when status is `succeeded`;
- first non-empty limitation when present;
- otherwise `"Coding agent could not complete the request."`.

`result_summary` rules:

- Include worker name, status, repository identity when available, short
  resolved commit/content identity when available, and evidence count.
- Cap to 500 characters.
- Do not include local paths, workspace paths, cache keys, raw excerpts, job
  ids, or delivery fields.

`worker_metadata.evidence_refs` must omit `excerpt`. The answer text may cite
repo-relative evidence produced by Phase 1, but metadata must not store source
excerpts.

## LLM Call And Context Budget

Use a conservative 50k-token effective context cap.

Before Phase 3:

- Live turn: existing cognition/L2d calls only.
- Background generic router: one `BACKGROUND_WORK_LLM` call.
- Text-artifact worker: zero to two `BACKGROUND_WORK_LLM` calls.
- Coding-agent direct interface: not connected to Kazusa runtime.

After Phase 3:

- Live turn: no additional LLM calls. L2d affordance wording changes, but L2d
  still emits only semantic action requests.
- Background generic router: still one `BACKGROUND_WORK_LLM` call. Payload
  includes task brief, source summary, max output chars, and enabled worker
  descriptions. Keep this payload under `BACKGROUND_WORK_INPUT_CHAR_LIMIT`.
- Coding worker: no extra model route. It calls the public standalone
  coding-agent interface selected for the job. PM/programmer/synthesis calls
  use the effective `CODING_AGENT_LLM` route or fallback `BACKGROUND_WORK_LLM`.
- Result-ready cognition: existing background-work result-ready cognition path.
  It receives bounded artifact text and prompt-safe summaries only.

Latency impact is background-only after queue acknowledgement. The live user
turn must not wait for repository fetching or reading.

## Change Surface

### Modify

- Plan registry/reference: `development_plans/README.md` and
  `development_plans/reference/designs/coding_agent_architecture.md`.
- Operator docs: `README.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/background_work/README.md`, and
  `src/kazusa_ai_chatbot/coding_agent/README.md`.
- Configuration: `src/kazusa_ai_chatbot/config.py` adds
  `CODING_AGENT_WORKSPACE_ROOT`.
- Background work core: `background_work/models.py`, `router.py`,
  `worker.py`, `providers.py`, `subagent/__init__.py`, and
  `subagent/text_artifact.py`.
- Action affordance only: `src/kazusa_ai_chatbot/action_spec/registry.py`
  updates prompt-safe wording while keeping schema route-only.
- Tests: `tests/test_background_work_router.py`,
  `tests/test_background_work_providers.py`,
  `tests/test_background_work_text_artifact.py`,
  `tests/test_background_work_delivery.py`,
  `tests/test_action_spec_evaluator.py`,
  `tests/test_l2d_action_selection_cases.py`,
  `tests/fixtures/l2d_background_artifact_cases.json`,
  `tests/test_l2d_action_selection_live_llm.py`, and
  `tests/test_background_work_router_live_llm.py`.

### Create

- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: background
  worker adapter over public standalone coding-agent interfaces.
- `tests/test_background_work_coding_agent.py`: focused deterministic worker
  adapter tests.
- `tests/test_background_work_coding_agent_live_llm.py`: one-case live/local
  LLM diagnostic for the Phase 3 path, marked `live_llm`.

### Keep

- Phase 0 `code_fetching` public contract.
- Phase 1 `answer_code_question(...)` public contract.
- Phase 2 standalone `code_writing` public contract when Phase 3 scope
  includes patch proposal work.
- `background_work_jobs` MongoDB schema version and collection.
- Existing result-ready cognition delivery module and service delivery
  boundary.
- Legacy `background_artifact` compatibility modules and old-row support.

## Overdesign Guardrail

- Actual problem: Kazusa cannot route code repository questions through its
  normal L2d/background-work path because the only current production worker is
  a text-artifact placeholder and the coding-agent direct interface is not
  registered as a background worker.
- Minimal change: add one `coding_agent` worker adapter, register it, update
  router normalization and affordance wording, pass trusted `task_brief` to
  workers, map Phase 1 responses into the existing `BackgroundWorkResult`, and
  add targeted tests.
- Ownership boundaries: L2d selects generic background work; action-spec code
  materializes queue fields; background-work router chooses worker; coding
  worker adapts Phase 1 direct interface; Phase 0 fetches; Phase 1 reads;
  result-ready cognition and L3/dialog own visible wording.
- Rejected complexity: no new queue, no direct service endpoint, no worker
  names in L2d, no deterministic keyword router, no code writing/execution,
  no workspace path from user text, no new LLM routes, no prompt projection of
  full worker metadata, no compatibility fallback from coding questions to
  text-artifact.
- Evidence threshold: add writing/execution, worker-local coding task
  classification, metadata prompt projection, or broader external help only
  after Phase 3 passes background-worker integration and a later approved plan
  names the new contract.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's contracts, cutover policies, and ownership
  boundaries.
- Do not introduce alternate migration strategies, compatibility layers,
  fallback paths, extra features, unrelated cleanup, dependency upgrades, broad
  prompt rewrites, or background-work database migration.
- Treat changes outside `background_work`, `coding_agent`, action-spec
  affordance wording, configuration, docs, and named tests as high-scrutiny; if
  another change is necessary, stop and update this plan first.
- Reuse existing helpers when equivalent behavior exists.
- If the plan and code disagree, preserve this plan's stated ownership
  boundary and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Confirm Phase 1 dependency.
   - Read the completed Phase 1 execution evidence, `coding_agent` README, and
     direct interface tests.
   - Run the Phase 1 focused direct tests named by the completed plan.
   - Expected result before Phase 3: selected standalone Phase 1 and Phase 2
     capabilities are completed and their direct interface tests pass.
2. Parent adds focused coding-worker tests.
   - File: `tests/test_background_work_coding_agent.py`.
   - Tests: successful response mapping, non-success mapping, missing
     `CODING_AGENT_WORKSPACE_ROOT`, max-output cap, metadata sanitization, and
     no absolute path leakage.
   - Expected before implementation: fails because worker module is missing.
3. Parent adds router/provider contract tests.
   - Files: `tests/test_background_work_router.py`,
     `tests/test_background_work_providers.py`.
   - Tests: router accepts enabled `coding_agent`, rejects unknown workers,
     route decision remains route-only, provider passes `task_brief` as
     deterministic worker context, and provider does not pass worker-local tool
     args.
   - Expected before implementation: fails because router is hardcoded to
     `text_artifact` and provider does not preserve `task_brief`.
4. Parent adds action-spec and L2d boundary tests.
   - Files: `tests/test_action_spec_evaluator.py`,
     `tests/test_l2d_action_selection_cases.py`, L2d fixture file.
   - Tests: prompt affordance stays worker-name-free, background-work schema
     stays route-only, and a codebase reading case requires `speak` plus
     `background_work_request`.
   - Expected before implementation: affordance wording is too narrow and the
     new fixture is absent.
5. Parent starts one production-code subagent.
   - Scope: production code and docs in `Change Surface`.
   - Inputs: this plan, mandatory skills, focused test failures, and exact
     contracts above.
   - Subagent must not edit tests unless the parent explicitly directs a
     review-fix correction.
6. Production-code subagent implements worker registration and routing.
   - Add `subagent/coding_agent.py`.
   - Register it in `subagent/__init__.py`.
   - Update router prompt and enabled-worker normalization.
   - Update providers/worker to pass `task_brief`.
   - Update text-artifact wording.
7. Production-code subagent implements configuration and docs.
   - Add `CODING_AGENT_WORKSPACE_ROOT`.
   - Update README/HOWTO/background-work/coding-agent ICDs.
8. Parent runs focused deterministic tests.
   - Run the tests added in steps 2 to 4.
   - If tests reveal contract gaps, update tests or implementation only to
     match this plan, then rerun.
9. Parent runs integration and regression tests.
   - Run focused background-work, action-spec, coding-agent, and L2d tests in
     the Verification section.
10. Parent runs live/local LLM diagnostics one case at a time.
    - Run router live case for `coding_agent`.
    - Run L2d live case for codebase reading route.
    - Run Phase 3 worker diagnostic for the real-demand question.
    - Inspect trace files and record judgment.
11. Parent starts one independent code-review subagent.
    - Review plan alignment, diff, test evidence, prompts, config, and
      sanitized metadata.
12. Parent remediates review findings inside approved scope.
    - Rerun affected tests and static checks.
    - Record final evidence before lifecycle updates or sign-off.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code and documentation
  changes only; does not edit tests unless the parent explicitly directs it;
  closes after planned production changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  live LLM diagnostics, and validation work while the production-code subagent
  edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 0 - Phase 1 dependency verified.
  - Covers: implementation step 1.
  - Verify/evidence: completed Phase 1 status plus focused direct test output
    recorded in `Execution Evidence`.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 1 - focused failing test contract established.
  - Covers: implementation steps 2 to 4.
  - Verify/evidence: new worker/router/task-brief tests fail before
    production implementation, with command output recorded.
  - Handoff: next agent starts production implementation at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - worker, router, provider, and config implementation complete.
  - Covers: implementation steps 5 to 7.
  - Verify/evidence: focused worker/router/provider/config tests pass and
    changed production files are recorded.
  - Handoff: next agent starts Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - integration and regression verification complete.
  - Covers: implementation steps 8 and 9.
  - Verify/evidence: background-work, action-spec, L2d deterministic,
    coding-agent, and static grep checks pass with command output recorded.
  - Handoff: next agent starts Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - live/local LLM diagnostics complete.
  - Covers: implementation step 10.
  - Verify/evidence: live LLM cases run one at a time; trace paths and quality
    judgments are recorded.
  - Handoff: next agent starts Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - independent code review complete.
  - Covers: implementation steps 11 and 12.
  - Verify/evidence: review approves or findings are remediated; rerun
    commands and residual risks are recorded.
  - Handoff: ready for lifecycle update and final sign-off.
  - Sign-off: `<agent/date>` after review evidence is recorded.

## Verification

### Static Greps

- `rg -n "worker.*coding_agent|coding_agent.*worker" src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/action_spec`
  - Expected: no matches in L2d runtime prompts, action-spec schemas, or
    prompt-safe affordances. Matches in tests are allowed only when asserting
    non-leakage.
- `rg -n "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM|CODING_AGENT_SYNTHESIZER_LLM|CODING_AGENT_WORKER_LLM" src tests docs README.md development_plans`
  - Expected: the only allowed matches are this plan's prohibition lines.
- `rg -n "local_root|workspace_root|cache_key" src/kazusa_ai_chatbot/background_work tests/test_background_work_coding_agent.py`
  - Expected: production matches only in sanitizer/prohibition tests or docs;
    no result mapping includes these fields.
- `rg -n "source_context.*or.*task_brief|source_summary = job.get\\(\"source_context\"" src/kazusa_ai_chatbot/background_work`
  - Expected: no match. Worker execution must not prefer source context over
    task brief.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_background_work_coding_agent.py -q`
- `venv\Scripts\python -m pytest tests\test_background_work_router.py tests\test_background_work_providers.py tests\test_background_work_text_artifact.py -q`
- `venv\Scripts\python -m pytest tests\test_background_work_jobs.py tests\test_background_work_delivery.py -q`
- `venv\Scripts\python -m pytest tests\test_action_spec_evaluator.py tests\test_l2d_action_selection_cases.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py -q`

### Config Tests

- Add and run `tests/test_background_work_coding_agent_config.py`.
- The test imports `kazusa_ai_chatbot.config` under patched environment values
  and verifies: absent `CODING_AGENT_WORKSPACE_ROOT` becomes `""`; present
  value is preserved; the coding worker fails closed when it is empty.

### Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\background_work\subagent\coding_agent.py src\kazusa_ai_chatbot\background_work\router.py src\kazusa_ai_chatbot\background_work\providers.py src\kazusa_ai_chatbot\background_work\worker.py`

### Live LLM Diagnostics

Run one case at a time with `-s`, inspect the trace, and record the trace path:

- `venv\Scripts\python -m pytest -m live_llm tests\test_background_work_router_live_llm.py::test_live_router_selects_coding_agent_for_repository_question -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_l2d_action_selection_live_llm.py::test_l2d_live_routes_repository_question_to_background_work -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests\test_background_work_coding_agent_live_llm.py::test_live_coding_agent_worker_answers_kazusa_image_reading_question -q -s`

The real-demand diagnostic passes only when the trace shows:

- L2d or direct test input preserves the repository/code question in
  `task_brief`;
- background-work router selects `coding_agent`;
- the worker passes configured `workspace_root` internally;
- `BackgroundWorkResult.artifact_text` answers from Phase 1 evidence;
- public result fields do not leak local paths, workspace roots, cache keys,
  raw command output, job ids, leases, or adapter ids.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread the architecture reference, completed Phase 0 plan,
completed Phase 1 and Phase 2 artifacts, this plan, and relevant
background-work/action source from a fresh-review posture.

Review scope:

- Previous and next phase handoffs are explicit: Phase 3 depends on completed
  standalone public interfaces from Phases 1 and 2, and Phase 3 keeps the same
  worker/queue boundary.
- L2d, action spec, router, worker, coding agent, result-ready cognition, and
  L3/dialog ownership remain separate.
- Contracts, files, verification gates, checklist, and evidence requirements
  are concrete enough for execution agents.
- No unresolved choices, broad verbs, fallbacks, shims, private helper freedom,
  or unowned side paths remain.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project style for changed Python, tests, prompts, docs, and commands.
- Design weaknesses: ownership leaks, hidden fallbacks, shims, prompt/context
  leaks, local path leaks, metadata leaks, deterministic keyword routing,
  persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, autonomy boundaries, change surface,
  contracts, implementation order, verification, and acceptance criteria.
- Regression and handoff quality, including Phase 1 and Phase 2 artifacts,
  Phase 3 notes, evidence, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows
review-only fixture/documentation corrections. If a fix would cross the
approved boundary or alter the contract, stop and update the plan or request
approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Selected standalone Phase 1 and Phase 2 capabilities are completed and
  accepted before Phase 3 production-code execution.
- `coding_agent` is registered as an enabled `background_work` worker alongside
  `text_artifact`.
- The background-work router can select `coding_agent` for repository/source
  code reading tasks and `text_artifact` for non-repository text artifacts.
- L2d/action-spec remains route-only and never exposes worker names,
  worker-local task types, repository paths, tool args, or final answer text.
- Worker dispatch passes trusted `task_brief` to workers and does not replace
  it with route reason/source context.
- The coding worker calls only public standalone coding-agent interfaces with
  configured `CODING_AGENT_WORKSPACE_ROOT` and max output cap.
- Missing `CODING_AGENT_WORKSPACE_ROOT` fails closed with a sanitized
  non-success result.
- Standalone coding-agent responses map into `BackgroundWorkResult` exactly as
  specified by their ICDs.
- Result-ready cognition and prompt payloads do not leak local paths,
  workspace roots, cache keys, raw source excerpts in metadata, raw command
  output, job ids, leases, adapter ids, or delivery fields.
- Text-artifact worker remains functional for snippets, rewrites, and
  summaries but no longer owns repository/codebase reading.
- All deterministic tests, static greps, compile checks, and one-case live LLM
  diagnostics in `Verification` pass and have recorded evidence.
- Independent code review approves the diff or all findings are remediated and
  affected verification is rerun.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Phase 3 integrates an unstable standalone contract | Block execution until selected Phase 1 and Phase 2 capabilities are completed and direct tests pass | Stage 0 dependency verification |
| Router overfits worker names or routes by local keywords | Use worker descriptions and LLM router; validate only enabled names deterministically | Router tests and no-prepost-user-input review |
| Worker loses the original code question URL | Pass `task_brief` separately to worker execution | Provider and worker tests |
| Local paths leak through metadata or result-ready cognition | Sanitize result mapping and omit excerpts from metadata | Sanitization tests and static greps |
| Text-artifact still handles repository reading | Tighten descriptions/prompts and route repository questions to coding worker | Router tests and live router case |
| Live chat latency increases | Keep fetching/reading entirely in background after queue acknowledgement | L2d/action-spec tests and service delivery boundary tests |

## Execution Evidence
- Record Phase 1 dependency, focused failing tests, changed files, static greps, deterministic tests, compile checks, live LLM traces, review, remediation, reruns, and residual risks:
