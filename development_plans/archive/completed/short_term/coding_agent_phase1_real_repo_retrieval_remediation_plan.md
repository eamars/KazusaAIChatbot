# coding agent phase1 real repo retrieval remediation plan

## Summary

- Goal: fix the Phase 1 code-reading agent's real-repository preparation,
  retrieval, and navigation weaknesses exposed by the `eamars/KazusaAIChatbot`
  image-response live LLM run, without source-shaped shortcuts or
  query-specific logic.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `debug-llm`, `no-prepost-user-input`,
  `superpowers:test-driven-development`, and `superpowers:executing-plans`.
- Overall cutover strategy: bigbang inside managed-checkout path internals and
  Phase 1 code-reading retrieval/planning internals; keep the public
  `answer_code_question(...)`, `code_fetching.run(...)`, and
  `code_reading.run(...)` interfaces.
- Highest-risk areas: cheating by encoding observed query/source terms,
  deterministic semantic routing over user input, managed-checkout paths that
  fail before reading on Windows, cleanup failures masking the primary clone
  error, weak lexical search on large repositories, local LLM context overload,
  noisy test/doc matches outranking implementation paths, and false confidence
  from synthetic fixtures.
- Acceptance criteria: six named real GitHub end-to-end gates prove the agent
  can prepare and navigate real-world repositories to grounded answers or
  narrow, evidence-specific clarification; prompt/code greps prove no
  observed-query/repo-specific terms are encoded in runtime prompts or
  deterministic matching logic; deterministic tests prove generic managed path
  robustness, retrieval ranking, scope safety, and report compaction;
  independent review finds no unresolved blockers.

## Context

Phase 1 passed synthetic live LLM gates, but the real GitHub run against
`eamars/KazusaAIChatbot` failed product-quality expectations. The execution
record is:

- Review artifact:
  `test_artifacts/llm_reviews/coding_agent_real_github_kazusa_image_response_review.md`
- Raw trace:
  `test_artifacts/llm_traces/coding_agent_real_github__kazusa_image_response__20260620T105248562175Z.json`
- Compact stage summary:
  `test_artifacts/llm_traces/coding_agent_real_github__kazusa_image_response__20260620T105248562175Z.stage_summary.json`

The important behavior was not a hallucination. It was a retrieval/navigation
failure:

1. The first Phase 0 attempt used the normal deep live-test workspace under
   `test_artifacts` and failed before reading because the managed checkout path
   was too deep for Windows/Git path handling.
2. The failed clone cleanup path also raised a filesystem cleanup error, which
   risked masking the primary checkout failure.
3. Phase 0 fetched the real GitHub repository successfully only after rerunning
   with a much shorter Windows workspace path.
4. The repository map exposed 938 safe files.
5. PM wave 1 asked for image-input handling and character-response generation.
6. Programmer workers found image-observation evidence but failed to find the
   downstream character-response path.
7. PM wave 2 refined image detection but shifted to a poor intent and again
   failed to locate response-generation evidence.
8. The supervisor hit the two-wave limit and returned `needs_user_input`
   instead of synthesizing an ungrounded answer.

The short-workspace rerun proves the later retrieval failure was real, but the
initial checkout failure is also a product failure. A code-reading agent must
first prepare a readable checkout robustly, then navigate from a broad but
ordinary question to relevant implementation paths in a large real repository.

## Failure Map

| Failure | Evidence | Root cause |
|---|---|---|
| Managed checkout path shape is too deep for Windows live-test workspaces | The first real GitHub attempt failed before Phase 1 until rerun from a short workspace | `managed_clone.build_managed_checkout_paths(...)` embeds provider, owner, repo, `refs`, ref key, and `checkout` in nested path components instead of using a compact hash-first layout. |
| Clone cleanup can mask the primary checkout failure | The failed checkout path also hit a cleanup filesystem error | `_clone_into_managed_checkout(...)` calls `shutil.rmtree(...)` during error handling without preserving the original Git failure as the public-safe primary error. |
| Flat lexical repository map is too weak for large repos | PM saw only a capped sorted file list and top directories for 938 safe files | The PM lacks a semantic file index, module summaries, and dependency/call hints needed to choose useful bounded scopes. |
| Search assignments are noisy | `char_resp_2` matched an unrelated control-console variable named `character_response_info` | Search rows are collected by substring scan without ranking by path role, symbol boundary, source-vs-test priority, or assignment intent. |
| Test files and docs can dominate implementation evidence | Successful image reports mostly came from tests and one architecture reference | Evidence collection does not separate implementation, tests, docs, plans, scripts, and config into typed source classes. |
| PM cannot ask for dependency-following work | PM can assign file, directory, symbol, or search only | There is no generic "follow imports/calls from current evidence" retrieval primitive, so the PM repeats searches instead of navigating. |
| Programmer reports do not compact candidate evidence into next-hop hints | PM wave 2 still searched broad terms instead of following from discovered identifiers | Reports include facts and evidence refs, but no generic discovered-symbol, candidate-file, or next-hop hint contract. |
| Two-wave limit protects context but cuts off large-repo exploration | Final result hit `reading_pm:sufficiency=wave_limit` | The current fanout/wave budget is adequate for synthetic fixtures, not for large real repos without better retrieval precision. |
| The real LLM test matrix underweighted large repos | Synthetic gates passed; real repo failed | The previous gates proved stage mechanics and grounding, but not retrieval robustness on noisy repositories. |

## Projected Future Failures

| Future failure class | Why it will happen if unaddressed | Expected symptom |
|---|---|---|
| Broad feature questions in large repos | File list and substring search do not expose architecture flow | `needs_user_input` after wave limit or irrelevant evidence from tests/docs. |
| Questions using product-language names instead of code identifiers | The agent lacks semantic module discovery and synonym expansion owned by the LLM | PM chooses generic search terms that miss implementation names. |
| Cross-layer questions | Current assignments inspect isolated slices but cannot follow adapters -> cognition -> dialog or controller -> service chains | Evidence covers one layer but synthesis cannot explain end-to-end behavior. |
| Noisy common terms | Terms such as response, request, message, image, handler, config, user, and action appear everywhere | Search rows point to unrelated tests, fixtures, or control-console code. |
| Test/docs overshadowing source | Tests often contain clearer labels than production code | Programmer facts become test-contract facts when the user asked for implementation behavior. |
| Checkout path-length failures before code reading | Deep caller workspaces plus nested managed checkout components leave too little path budget for real repository files | Phase 0 fails before reading unless managed path construction is compact and validates path budget before clone. |
| Failed clone cleanup leaves temp dirs or masks root cause | Git failures on Windows can leave locked pack files or overlong paths that resist immediate deletion | Caller sees a secondary cleanup exception instead of a structured checkout failure and may leave stale temp state. |
| False positive grounding | A final answer may be grounded to docs or tests but not implementation | Answer is evidence-backed but unsatisfactory because evidence class is wrong for the question. |
| False negative refusal | Enough evidence exists in the repo, but retrieval cannot find it | Agent conservatively refuses, which is safe but fails the product goal. |

## Mandatory Skills

- `development-plan`: load before editing this plan, executing it, updating
  lifecycle records, or reviewing completion.
- `local-llm-architecture`: load before changing PM/programmer contracts,
  retrieval primitives, prompt surfaces, LLM context budgets, or synthesis.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running or reviewing live LLM diagnostics.
- `no-prepost-user-input`: load before designing any user-input interpretation
  or semantic routing behavior; do not add deterministic user-input keyword
  routing.
- `superpowers:test-driven-development`: load before production
  implementation; parent must establish failing or baseline tests first.
- `superpowers:executing-plans`: load before execution; user has explicitly
  instructed parent-only fallback execution with no subagents for this plan.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must stop before the plan's `Independent Code Review` gate and
  get user direction on a no-subagent review mechanism.
- The plan's `Execution Model` must use the user-approved parent-only fallback
  path with no subagents.
- Do not encode any gate user request, gate repository owner/repo,
  gate-specific wording, source-specific answer path, or discovered
  source-specific answer fact into runtime prompts, deterministic matching
  code, routing constants, ranking boosts, fallback logic, or production
  examples.
- Do not add deterministic keyword routing over the user's question. The LLM PM
  owns semantic interpretation; deterministic code may rank repository
  artifacts and evidence rows using generic file metadata, lexical match
  quality, symbol boundaries, path class, import/call relationships, caps, and
  safety rules.
- Prompt changes must be generic and reusable. They may define capability
  contracts such as "prefer implementation evidence when the user asks how code
  works"; they must not mention any gate repo, gate feature, gate nouns, trace
  artifact, or expected answer.
- Runtime tests may use the observed real GitHub case as a regression gate, but
  passing it alone is not acceptance. All six named real-world gates in this
  plan must pass the same generic retrieval rubric.
- The six gate questions, repository identities, and any source paths or answer
  facts discovered while reviewing those gates are test/review artifacts only.
  They must not appear in production prompts, deterministic matching code,
  ranking boosts, routing constants, fallback logic, or production examples.
- Do not increase context by sending whole repositories, raw unbounded search
  dumps, or full source files to the PM or synthesis stage.
- Do not add web research, external code search, package execution, Docker,
  test execution against fetched repositories, or patch-writing behavior.

## Must Do

- Record the observed failure as a blocker for Phase 1 final sign-off until
  this remediation or an equivalent approved plan is executed.
- Fix managed checkout path construction so Phase 0 uses compact, bounded,
  hash-first internal paths and no longer depends on long owner/repo/ref path
  components for storage identity.
- Fix failed managed-clone cleanup so cleanup is best effort, preserves the
  primary Git/path failure, and returns a public-safe structured checkout
  limitation instead of leaking local roots or raising a secondary cleanup
  exception.
- Add a generic repository-intelligence layer that exposes safe, compact,
  source-class-aware file/module summaries to the PM.
- Replace flat substring evidence collection for search assignments with a
  generic ranked retrieval contract that distinguishes implementation, tests,
  docs, plans, scripts, and config.
- Add a generic dependency/navigation retrieval primitive that can follow from
  discovered evidence to related imports, callers, definitions, and adjacent
  modules without user-query-specific rules.
- Extend programmer reports with generic discovered-symbol and candidate-next
  evidence hints, while preserving the simplified PM/programmer ownership
  boundary.
- Keep supervisor-owned caps, but make the budget apply to ranked candidates
  and next-hop evidence rather than raw first-matches.
- Add real LLM gates for the observed failure plus five additional real-world
  scenarios without source-shaped expected-answer code.
- Replace the earlier simple live LLM gates with the six named real-world gates
  in this plan. These six gates block Phase 1 sign-off.
- Add static greps that fail if any gate repository identity, full gate
  question text, source-specific answer path, or discovered source-specific
  answer fact appears in production prompts or deterministic code.
- Update debug review artifacts with full stage traces and human judgments for
  every real LLM gate.

## Deferred

- Do not integrate Phase 1 with Kazusa service, adapters, background workers,
  L2d, dialog delivery, persistence, scheduler, or dispatcher.
- Do not implement code writing, patch proposal, patch application, package
  installation, shell execution, Docker execution, or fetched-repo test runs.
- Do not add separate PM, programmer, or synthesizer model routes.
- Do not add a broad multi-agent distributed master-PM architecture unless a
  later approved plan expands Phase 1 beyond standalone code reading.
- Do not add source-specific examples or expected answer phrases to runtime
  prompts.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Managed checkout internals | bigbang | Replace nested owner/repo/ref storage paths with compact hash-first paths. Store human identity in metadata, not path depth. Do not preserve the old path shape as fallback. |
| Phase 0 public result contract | compatible | Keep `code_fetching.run(...)` result shape and public fields. Checkout failures must stay public-safe and must not expose local roots, workspace roots, or cache keys. |
| Retrieval internals | bigbang | Replace flat first-match search behavior with ranked, source-class-aware retrieval. Do not preserve old search ordering as fallback. |
| Public interface | compatible | Keep `answer_code_question(...)` and `code_reading.run(...)` response shapes. |
| PM/programmer IO | compatible | Preserve `PMInput`, `PMDecision`, `ProgrammerAssignment`, and `ProgrammerReport`, but extend reports only with generic optional fields if tests prove they are needed. |
| Prompts | bigbang | Rewrite prompt contract text only where necessary for generic retrieval ownership. No source-shaped examples. |
| Tests | bigbang | Add real large-repo gates and replace weak retrieval assertions. Do not make deterministic tests assert the observed answer. |

## Target State

For large repositories, Phase 1 should follow this bounded path:

```text
CodeReadingRequest
-> Phase 0 managed checkout path planning
-> compact checkout succeeds or returns a structured public-safe checkout limitation
-> Phase 0 repository/source scope
-> repository intelligence summary
-> PM chooses generic evidence needs
-> ranked retrieval selects candidate files/symbols
-> programmer reports facts plus generic next-hop hints
-> PM requests follow-up bounded retrieval when needed
-> PM declares sufficient only when answer slots are source-backed
-> synthesis answers from selected evidence or asks for a specific narrower scope
```

When evidence is incomplete, the final limitation must identify the missing
generic evidence class, such as "the response-generation path was not found
from the selected candidates", rather than a vague wave-limit message.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Anti-cheating boundary | Forbid observed request/repo/source terms in prompts and deterministic code | Fixes must generalize beyond the failing case. |
| User query interpretation | Keep semantic decomposition in PM LLM | Deterministic keyword routing over user input violates project architecture. |
| Managed checkout paths | Use compact hash-first storage paths and metadata-owned identity | Repository owner, repo, and ref names are useful metadata but must not create deep storage paths that fail before reading. |
| Clone cleanup | Treat cleanup as best effort and preserve the primary failure | Cleanup errors should be visible in trace/debug evidence but must not replace the checkout failure seen by the caller. |
| Retrieval ranking | Deterministic code ranks repository artifacts, not user intent | File metadata, source class, symbol boundaries, and graph links are deterministic repository facts. |
| Source classes | Classify files as implementation, tests, docs, plans, scripts, config, generated/other | Prevents tests/docs from accidentally masquerading as implementation evidence. |
| Follow-up retrieval | Add generic dependency/call/import navigation | Large repo questions often require moving from discovered symbols to adjacent modules. |
| Context control | Rank before LLM, summarize before PM, cap every stage | Local LLMs need compact semantic inputs. |
| Acceptance | Real LLM traces are primary | Deterministic ranking tests cannot prove the model uses evidence well. |

## Contracts And Data Shapes

Existing public `CodeReadingRequest` and `CodeReadingResult` remain unchanged.

Existing public `code_fetching.run(...)` result shape remains unchanged.
Internal `ManagedCheckoutPaths` may change only behind the code-fetching
boundary. The path contract is:

```python
{
    "workspace_root": str,
    "ref_key": str,
    "checkout_root": str,    # compact internal path under workspace root
    "metadata_path": str,    # metadata stores provider/owner/repo/ref identity
    "temporary_root": str,   # compact temp path under workspace root
    "lock_path": str,        # compact lock path under workspace root
    "cache_key": str,        # stable non-secret cache identity
}
```

The managed path layout must keep generated path suffixes short, must remain
inside `workspace_root`, and must not use owner/repo/ref slug nesting as the
storage path. If the caller-provided workspace root is itself too deep for a
conservative Windows path budget, Phase 0 must fail before clone with a
structured public-safe limitation that asks for a shorter configured coding
workspace root.

Repository intelligence summary adds generic model-facing metadata:

```python
{
    "total_safe_files": int,
    "source_classes": {
        "implementation": list[FileSummary],
        "tests": list[FileSummary],
        "docs": list[FileSummary],
        "plans": list[FileSummary],
        "scripts": list[FileSummary],
        "config": list[FileSummary],
    },
    "top_symbols": list[SymbolSummary],
    "directories": list[DirectorySummary],
}
```

`FileSummary` must stay compact and generic:

```python
{
    "path": str,
    "source_class": str,
    "module_tokens": list[str],
    "defined_symbols": list[str],
    "imported_modules": list[str],
    "short_excerpt": str,
}
```

If `ProgrammerReport` is extended, only generic optional fields are allowed:

```python
{
    "discovered_symbols": list[str],
    "candidate_next_hops": list[{
        "reason": str,
        "scope": {"kind": "file | directory | symbol | search", "values": list[str]},
    }],
}
```

These fields must contain repository-discovered terms, not hardcoded terms from
the observed failure.

## LLM Call And Context Budget

Use split code-reading routes. PM decisions and final synthesis use
`CODING_AGENT_PM_LLM`; programmer workers use
`CODING_AGENT_PROGRAMMER_LLM`. The previous shared route is retired by
big-bang cutover and must not remain in runtime code, docs, or tests.

| Stage | Before | After | Context policy |
|---|---:|---:|---|
| Repository intelligence | 0 LLM calls | 0 LLM calls | Deterministic AST/text metadata only. |
| PM decision | 1 call per wave on the shared code-reading route | 1 call per wave on `CODING_AGENT_PM_LLM` | PM sees compact source-class and candidate summaries, not raw file lists or full search dumps. |
| Programmer | 1 call per assignment on the shared code-reading route | 1 call per assignment on `CODING_AGENT_PROGRAMMER_LLM` | Programmer sees ranked bounded evidence rows plus source-class labels. |
| Synthesis | 1 call when sufficient on the shared code-reading route | 1 call on `CODING_AGENT_PM_LLM` | Synthesis sees selected evidence rows and limitations only. |

The normal maximum is three PM waves with the existing six-report total cap.
This cap increase is justified by Gate 3 traces where ranked retrieval found
relevant implementation evidence but the second wave still ended with
no-evidence follow-up assignments and false final sufficiency. Do not increase
the cap beyond three waves without a new approved plan or explicit user
approval.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_clone.py`: replace
  nested owner/repo/ref managed paths with compact hash-first checkout, temp,
  metadata, and lock paths; validate path budget before clone; make clone
  cleanup best effort while preserving the primary failure.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py`: ensure managed
  checkout failures remain structured and public-safe, without leaking absolute
  local paths, workspace roots, or cache keys.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md`: document the
  compact managed path policy and public-safe failure behavior.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/repository_map.py`: replace
  flat capped file list with compact source-class-aware repository
  intelligence.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/evidence.py`: replace
  first-match substring search with ranked candidate retrieval and generic
  next-hop evidence selection.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/models.py`: add generic
  optional report metadata only if implementation requires it.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/product_manager.py`: update
  PM prompt and normalization only for generic source-class and follow-up
  retrieval contracts.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/programmer.py`: include
  source-class labels and optional next-hop hints in report payloads.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/supervisor.py`: preserve
  caps while passing generic repository intelligence and improved limitations.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md`: document
  retrieval ranking, source classes, and failure modes.
- `tests/test_coding_agent_reading*.py`: add deterministic support tests for
  ranking, source classes, next-hop evidence, caps, and anti-cheating greps.
- `tests/test_coding_agent_fetching.py`: add deterministic tests for compact
  managed checkout path construction, path-budget validation, and cleanup-error
  preservation on clone failure.
- `tests/test_coding_agent_live_llm.py`: add one-case-per-test real LLM gates
  for large/noisy repository navigation and PM-thinking/programmer-default
  comparison.
- `src/kazusa_ai_chatbot/config.py`: replace the shared code-reading LLM
  route with full PM and programmer route identities.
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`: report PM and
  programmer code-reading routes separately.
- `test_artifacts/llm_reviews/`: add or update agent-authored review artifacts
  for new real LLM gates.

### Create

- A focused deterministic fixture repository with noisy tests/docs/config and a
  differently named implementation flow for retrieval ranking tests.
- A managed-checkout path regression fixture or monkeypatched clone test that
  reproduces the long-workspace failure class without relying on a specific
  GitHub repository name.
- A real LLM trace review artifact for the observed real GitHub failure rerun
  after remediation.

### Keep

- Public Phase 0 fetching result contract.
- Public direct interface response shape.
- Read-only Phase 1 boundary.

## Overdesign Guardrail

- Actual problem: Phase 1 can answer small synthetic fixtures but can fail
  before reading on normal Windows live-test paths and cannot reliably navigate
  large noisy repositories to source-backed implementation answers.
- Minimal change: compact managed checkout paths and improve generic
  repository intelligence, ranked retrieval, and follow-up evidence navigation
  while keeping PM/programmer/synthesis and public interfaces intact.
- Ownership boundaries: PM LLM owns semantic decomposition; deterministic code
  owns repository metadata extraction, managed path safety, cleanup behavior,
  source-class labeling, ranking mechanics, caps, and grounding; programmers
  own source fact extraction; synthesis owns final explanation from selected
  evidence.
- Rejected complexity: no query-specific code, no source-shaped prompts, no web
  search, no execution against fetched repos, no service integration, no
  separate model routes, no unlimited wave count, no whole-repo prompt.
- Evidence threshold: only after real large-repo gates still fail with ranked
  retrieval may a later plan consider additional architecture such as a
  dedicated repository-index artifact or broader multi-agent exploration.

## Agent Autonomy Boundaries

- The responsible agent may choose local ranking mechanics only when they use
  generic repository facts and do not inspect or special-case the observed user
  query or observed repo.
- The responsible agent must not introduce new runtime prompt examples that
  contain observed failure terms or expected answers.
- The responsible agent must not add deterministic user-query keyword routing,
  semantic post-processing of PM decisions, or code that rewrites LLM semantic
  intent based on local string matching.
- The responsible agent must treat every prompt edit as high scrutiny and run
  anti-cheating greps before claiming completion.
- If the observed real GitHub gate passes but another large-repo gate fails in
  the same failure class, the work is not complete.
- If implementation appears to require expanding Phase 1 into service
  integration or code execution, stop and request a new plan.

## Implementation Order

1. Parent adds failing/baseline tests and trace gates.
   - Add managed-checkout path and cleanup regression tests before retrieval
     work, covering compact path construction and cleanup-error preservation.
   - Add deterministic retrieval-ranking tests with noisy source classes.
   - Add real LLM gates for all six named real-world cases in this plan.
   - Record current baseline failures or insufficient behavior.
2. Parent adds anti-cheating static gates.
   - Grep production prompts and deterministic code for all gate repository
     identities, full gate question text, source-specific answer paths, and
     discovered source-specific answer facts.
   - Expected before implementation: no new violations.
3. Parent implements production changes directly under the user-approved
   no-subagent fallback path.
   - Parent must keep production changes inside this plan's change surface.
   - Parent must not use live test expected judgments as production logic.
4. Parent implements compact managed checkout paths, repository intelligence,
   and ranked retrieval after failing/baseline tests are established.
5. Parent runs deterministic retrieval, scope-safety, anti-cheating, and
   regression tests.
6. Parent runs each real LLM gate one at a time and writes debug review notes.
7. Parent updates documentation and execution evidence.
8. Parent stops before code review and asks the user how to proceed with a
   no-subagent review gate.
9. Parent fixes later review findings only after the user approves the review
   mechanism and only inside this plan's change surface.

## Execution Model

- Parent agent owns orchestration, test code, production code, real LLM
  validation, static checks, execution evidence, lifecycle updates, and final
  sign-off.
- Parent agent establishes the focused failing/baseline test contract first.
- Parent-only fallback is explicitly approved by the user for this plan.
- No subagents may be spawned or used for production implementation,
  verification, real LLM judgment, or code review in this plan execution.
- Parent must stop before starting any code review and ask the user how to
  proceed with review under the no-subagent constraint.

## Progress Checklist

- [x] Stage 1 - failure contract and anti-cheating gates established
  - Covers: implementation steps 1-2.
  - Verify: baseline traces, managed-checkout path tests, and deterministic
    retrieval tests record the current Phase 0 and large-repo failures;
    anti-cheating greps are defined.
  - Evidence: record commands, trace paths, and baseline judgments.
  - Handoff: next execution resumes at Stage 2.
  - Sign-off: completed 2026-06-20.
- [x] Stage 2 - compact checkout and generic retrieval architecture implemented
  - Covers: implementation steps 3-4.
  - Verify: focused deterministic managed-checkout, retrieval, source-class,
    next-hop, and contract tests pass.
  - Evidence: record changed files and focused test output.
  - Handoff: next execution resumes at Stage 3.
  - Sign-off: completed 2026-06-20.
- [x] Stage 3 - real LLM real-world gates pass by judgment
  - Covers: implementation steps 5-7.
  - Verify: every real LLM case runs one at a time, writes raw trace evidence,
    and receives agent-authored quality judgment.
  - Evidence: review artifact paths, trace paths, pass/fail judgments, and
    residual risks.
  - Handoff: next execution resumes at Stage 4.
  - Sign-off: completed 2026-06-20.
- [x] Stage 3b - code-reading LLM route split complete
  - Covers: user-approved 2026-06-21 big-bang demand.
  - Verify: focused route/config tests pass; grep proves the retired shared
    route is absent from `src`, `tests`, `docs`, and `README.md`; comparison
    live LLM artifact includes PM-thinking/programmer-default run.
  - Evidence: red/green route tests, static grep, real LLM trace, and review
    artifact.
  - Handoff: next execution resumes at Stage 4.
  - Sign-off: completed 2026-06-21.
- [x] Stage 4 - code review completed
  - Covers: implementation steps 8-9.
  - Verify: parent stopped before code review, received explicit user
    direction for a no-subagent review mechanism, executed the review in the
    parent agent, fixed surfaced blockers, and reran focused verification.
  - Evidence: user direction, reviewer findings, fixes, rerun commands, and
    approval status are recorded in `Execution Evidence`.
  - Handoff: remediation plan is ready for completed-plan lifecycle update.
  - Sign-off: completed 2026-06-21.

## Verification

### Static Greps

- Grep production prompts and deterministic code for all six gate repository
  identities, full gate question text, source-specific answer paths, and
  discovered source-specific answer facts. Expected: no matches outside test
  artifacts, review artifacts, this plan, and explicitly named test cases.
- Retired shared code-reading route token grep over `src tests docs README.md`
  - Expected: no matches. The shared route is retired and must not remain in
    runtime code, tests, or public docs.
- `rg -n "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM" src tests docs README.md`
  - Expected: matches only for the new PM and programmer route contracts.
- Separate synthesizer-route token grep over `src tests docs README.md`
  - Expected: no matches. Synthesis reuses `CODING_AGENT_PM_LLM`.
- Grep for whole-repo prompt/dump behavior in `code_reading`. Expected: no new
  full-file or full-repository prompt path.

### Deterministic Tests

- Focused retrieval ranking tests for:
  - implementation files outranking unrelated tests/docs for implementation
    questions;
  - tests/docs remaining available when the question asks for tests/docs;
  - symbol-boundary matches outranking substring-only matches;
  - next-hop candidates from imports/calls;
  - caps preserving source diversity instead of first-match ordering.
- Managed checkout tests for:
  - compact hash-first checkout, temp, metadata, and lock paths under the
    configured workspace root;
  - absence of owner/repo/ref slug nesting in storage path depth;
  - conservative Windows path-budget validation before clone when the configured
    workspace root is too deep;
  - clone failure cleanup preserving the original managed-clone error even when
    temp cleanup also fails;
  - public-safe Phase 0 failure results that do not expose absolute local roots,
    workspace roots, cache keys, or raw Git command output.
- Existing coding-agent deterministic regression:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py -q`
- Code-reading route split:
  `venv\Scripts\python -m pytest tests\test_coding_agent_reading_pm_programmer.py::test_code_reading_llm_routes_require_full_pm_and_programmer_settings tests\test_coding_agent_reading_pm_programmer.py::test_code_reading_pm_route_is_reused_for_synthesis tests\test_coding_agent_reading_pm_programmer.py::test_code_reading_llm_route_partial_configuration_fails_fast tests\test_llm_interface_route_report.py::test_llm_route_inventory_contains_all_routes_once tests\test_llm_interface_route_report.py::test_llm_route_inventory_uses_configured_models_and_sources tests\test_config.py::TestRouteLlmConfig::test_generic_chat_llm_config_is_removed tests\test_config.py::TestRouteLlmConfig::test_all_route_config_values_are_present -q`

### Real LLM Gates

Run one case at a time with `-q -s`, inspect each trace, and update an
agent-authored review artifact before continuing.

Required real LLM gates:

- Gate 1 - Character image response:
  - Repository: `https://github.com/eamars/KazusaAIChatbot`
  - Question: "Per https://github.com/eamars/KazusaAIChatbot project, how does
    the character respond to the image."
  - This gate must run from the normal deep live-test workspace path, or an
    equivalently deep Windows workspace path, to prove Phase 0 no longer
    depends on manual short-path reruns.
- Gate 2 - Home Assistant entity action flow:
  - Repository: `https://github.com/home-assistant/core`
  - Question: "Per https://github.com/home-assistant/core, when a user turns
    on a light entity from the UI or API, how does Home Assistant route that
    request to the integration, update the state machine, and broadcast the
    resulting state change?"
- Gate 3 - Zulip notification decision flow:
  - Repository: `https://github.com/zulip/zulip`
  - Question: "Per https://github.com/zulip/zulip, when a stream message
    mentions a user who may have muted the stream or topic and may be offline,
    how does Zulip decide whether that user gets an unread mention, desktop
    notification, mobile push, or email?"
- Gate 4 - Airflow task and DAG state flow:
  - Repository: `https://github.com/apache/airflow`
  - Question: "Per https://github.com/apache/airflow, when a task instance
    finishes and the DAG run contains a mix of successful, skipped, failed, and
    upstream-failed tasks, how does Airflow decide what downstream tasks to
    schedule next and when to mark the DAG run finished?"
- Gate 5 - FastAPI generator dependency lifecycle:
  - Repository: `https://github.com/fastapi/fastapi`
  - Question: "Per https://github.com/fastapi/fastapi, if an endpoint uses a
    generator-style dependency and the endpoint raises an HTTP exception before
    returning normally, what happens to the dependency cleanup code, exception
    handling, and final response sent to the client?"
- Gate 6 - ComfyUI uploaded image execution flow:
  - Repository: `https://github.com/comfy-org/comfyui`
  - Question: "Per https://github.com/comfy-org/comfyui, when a user uploads
    an image for a workflow and queues execution, how does ComfyUI move that
    image from upload/storage into node execution and then report progress or
    output back to the browser?"

The previous simple gates are discarded. These six gates are blocking: Phase 1
cannot be signed off until all six pass by human review of the real LLM traces.
Do not add deterministic user-query keyword routing, repo-specific boosts, or
pre-fabricated knowledge for these questions. The production code must use only
generic repository metadata, source-class labels, lexical quality, symbol
boundaries, import/call relationships, caps, validation, and LLM-owned semantic
decomposition.

Acceptance for these gates is based on human review of real traces, not pytest
green alone.

#### Gate Pass Condition

Each of the six real-world gates passes only when an AI agent reviews the real
LLM trace and writes an agent-authored review artifact that explicitly marks
the gate as passed. Deterministic keyword matching, substring matching, fixture
labels, pytest status, or schema success must not decide pass/fail.

A gate is a failure if any of these conditions occur:

- Environment failure: repository checkout, path handling, network access,
  local LLM availability, timeout, test harness execution, trace capture, or
  filesystem cleanup fails. The case may be rerun after the environment or
  implementation is fixed, but the failed run cannot count as a pass.
- Failure to get an answer: the final result is `needs_user_input`, a generic
  refusal, no synthesis, an empty answer, or a vague "cannot determine" answer
  when the repository is expected to contain the implementation evidence.
- Unrelated answer: the final answer discusses adjacent concepts, docs, tests,
  or a different subsystem without answering the concrete real-world behavior
  asked by the gate.
- Wrong workflow or wrong owner: the result is produced by manual source
  inspection, a test harness shortcut, precomputed knowledge, direct source
  lookup outside the coding-agent path, or any agent other than the intended
  Phase 0 fetching -> PM -> programmer -> synthesis workflow.
- Loop-limit incompletion: the supervisor hits a wave, loop, budget, or context
  limit before producing a complete evidence-grounded answer.

The reviewing AI agent must judge whether the answer covers the complete
logical chain in the gate question, whether the evidence comes from the correct
implementation source class, and whether the trace shows the intended
coding-agent workflow. The review may cite concrete evidence paths discovered
by the run, but those discovered paths remain test/review artifacts and must
not be copied into production prompts or deterministic logic.

### Compile

- `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`

## Independent Code Review

Do not run this gate during the current execution without further user
direction. The user has explicitly instructed that no subagents be used for
this plan. After all non-review verification commands pass, the parent must
stop and ask the user how to proceed with a no-subagent review mechanism.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and artifact.
- Anti-cheating compliance: no gate query, repo, source-specific answer, or
  regression-specific wording appears in runtime prompts or deterministic code.
- Architecture alignment: PM owns semantic decomposition; deterministic code
  owns retrieval mechanics and validation; programmers own source facts;
  synthesis owns evidence-grounded explanation.
- Retrieval quality: source-class ranking, next-hop evidence, caps, and
  limitations are generic and not source-shaped.
- Real LLM evidence quality: each real-world gate ran one at a time, was
  inspected, and has a review artifact.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The observed real GitHub failure no longer ends in a generic wave-limit
  limitation when sufficient implementation evidence exists.
- The observed real GitHub case no longer requires a manual short workspace
  path. From the normal deep live-test workspace path, Phase 0 must succeed
  with compact managed paths. A path-budget, checkout, clone, or cleanup
  failure is an environment/implementation failure and does not pass the gate.
- Managed clone failures preserve the primary Git/path error and do not surface
  secondary cleanup exceptions as the caller-visible failure.
- All six named real-world gates pass by real LLM trace judgment without
  source-shaped prompts, deterministic user-query keyword routing, repo-specific
  boosts, or pre-fabricated production knowledge.
- Each real-world gate pass is marked by an AI-agent-authored review artifact,
  not deterministic keyword matching or pytest success alone.
- Environment failures, failure to get an answer, unrelated answers, wrong
  workflow/agent ownership, and incomplete results due to loop or budget limits
  are failures for the six gate cases.
- For any non-gate negative regression case retained outside the six positive
  gates, a genuinely unsupported question returns a specific missing-evidence
  limitation naming the missing generic evidence class.
- Static anti-cheating greps show no gate-specific terms in production
  prompts or deterministic code.
- Deterministic retrieval tests prove generic source-class ranking, symbol
  boundary ranking, next-hop evidence selection, caps, and path safety.
- Public responses remain grounded to selected evidence rows and do not expose
  local roots, workspace roots, cache keys, secret-like files, raw source dumps,
  or binary content.
- Existing Phase 1 live LLM gates continue to pass one at a time by trace
  judgment.
- User-approved no-subagent code review has no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Phase 0 still fails before reading on Windows | Compact hash-first paths, pre-clone path-budget validation, and deep-workspace real gate | Managed checkout deterministic tests and observed real GitHub rerun from deep workspace |
| Cleanup masks checkout root cause | Best-effort cleanup helper preserves primary exception and records cleanup failure only as secondary trace/debug detail | Monkeypatched clone-failure cleanup test |
| Fix overfits the gate questions | Anti-cheating greps and all six real-world gates | Static greps and review artifact inspection |
| Ranking hides useful tests/docs | Source class is a ranking signal, not a hard exclusion | Tests where user asks for test/doc behavior |
| Local LLM context grows too large | Deterministic ranking before prompt input and strict caps | Context-size trace review |
| PM repeats weak searches | Programmer reports include generic next-hop candidates | Real trace shows PM follows candidates |
| Evidence is grounded but wrong class | Evidence rows carry source class and synthesis states evidence class | Review artifact checks source class fit |
| Wave limit remains too strict | Improve retrieval first; do not raise wave count unless traces justify it | Large-repo gates after ranked retrieval |

## Execution Evidence

Record execution evidence here during implementation. Do not pre-fill checked
boxes or success claims before commands are run.

- Draft created from real failure artifact:
  `test_artifacts/llm_reviews/coding_agent_real_github_kazusa_image_response_review.md`
- Stage 1/2 focused deterministic baseline initially failed for compact checkout
  path depth, path-budget rejection, cleanup failure preservation, repository
  source classification, implementation-first evidence ranking, and expanded
  programmer report shape.
- Focused deterministic remediation rerun passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py::test_managed_checkout_paths_use_compact_hash_storage tests\test_coding_agent_fetching.py::test_managed_checkout_rejects_workspace_root_with_no_path_budget tests\test_coding_agent_fetching.py::test_managed_clone_preserves_git_failure_when_cleanup_fails tests\test_coding_agent_reading.py::test_repository_intelligence_classifies_sources_and_symbols tests\test_coding_agent_reading.py::test_search_evidence_prefers_implementation_over_tests_and_docs tests\test_coding_agent_reading_pm_programmer.py::test_contracts_define_simplified_pm_programmer_shapes tests\test_coding_agent_reading_pm_programmer.py::test_programmer_report_uses_simplified_memory_shape -q`
  - Result: 7 passed.
- Deterministic plan suite initially exposed one regression: raw GitHub file
  download incorrectly inherited the clone checkout path-budget guard.
  Root cause: raw download storage reused managed checkout path construction
  even though raw downloads do not create deep git working trees.
  Remediation: keep clone checkout path-budget enforcement enabled by default
  and opt out only for managed raw-file downloads.
- Raw GitHub file regression check passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py::test_run_downloads_raw_github_file_without_clone -q`
  - Result: 1 passed.
- Deterministic plan suite passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py -q`
  - Result: 61 passed.
- Anti-cheat static greps:
  - `rg -n "eamars/KazusaAIChatbot|home-assistant/core|zulip/zulip|apache/airflow|fastapi/fastapi|comfy-org/comfyui|character respond to the image|light entity|stream message|task instance|generator-style dependency|uploads an image" src\kazusa_ai_chatbot\coding_agent`
    - Result: no production matches.
  - `rg -n "full repository|whole repository|full source|raw source|all files|entire repository" src\kazusa_ai_chatbot\coding_agent\code_reading`
    - Result: existing prohibition text only in prompts/README.
- Stage 3 real LLM gates passed by agent-authored trace review:
  `test_artifacts/llm_reviews/coding_agent_phase1_hard_gates_review.md`
  - Gate 1 Character image response:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_character_image_response__20260620T115853008438Z.json`
  - Gate 2 Home Assistant entity action flow:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_home_assistant_entity_action__20260620T121210948002Z.json`
  - Gate 3 Zulip notification decision flow:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_zulip_notification_decision__20260620T125247915755Z.json`
  - Gate 4 Airflow task and DAG state flow:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_airflow_task_state_flow__20260620T134116495633Z.json`
  - Gate 5 FastAPI generator dependency lifecycle:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_fastapi_dependency_lifecycle__20260620T140020196374Z.json`
  - Gate 6 ComfyUI uploaded image execution flow:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_comfyui_uploaded_image_flow__20260620T141332298151Z.json`
- Stage 3b route split TDD baseline failed as expected before production
  migration:
  - Focused route resolver tests failed with missing
    `resolve_code_reading_llm_settings`.
  - Route inventory/config tests failed because startup diagnostics still
    exposed the retired shared code-reading route and did not expose the new
    PM/programmer routes.
- Stage 3b focused route split verification passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_reading_pm_programmer.py::test_code_reading_llm_routes_require_full_pm_and_programmer_settings tests\test_coding_agent_reading_pm_programmer.py::test_code_reading_pm_route_is_reused_for_synthesis tests\test_coding_agent_reading_pm_programmer.py::test_code_reading_llm_route_partial_configuration_fails_fast tests\test_llm_interface_route_report.py::test_llm_route_inventory_contains_all_routes_once tests\test_llm_interface_route_report.py::test_llm_route_inventory_uses_configured_models_and_sources tests\test_config.py::TestRouteLlmConfig::test_generic_chat_llm_config_is_removed tests\test_config.py::TestRouteLlmConfig::test_all_route_config_values_are_present -q`
  - Result: 7 passed in 0.78s.
- Stage 3b broader route/config verification passed:
  `venv\Scripts\python -m pytest tests\test_config.py tests\test_llm_interface_route_report.py tests\test_coding_agent_reading_pm_programmer.py -q`
  - Result: 71 passed in 5.98s.
- Stage 3b static route greps:
  - Retired shared route token grep over `src tests docs README.md`:
    no matches; `rg` exit code 1.
  - `rg -n "CODING_AGENT_PM_LLM|CODING_AGENT_PROGRAMMER_LLM" src tests docs README.md`
    matched only the new PM/programmer route contracts.
  - Separate synthesizer-route token grep over `src tests docs README.md`:
    no matches; `rg` exit code 1.
- Stage 3b Character Image Response live comparison passed with PM thinking on
  and programmer thinking off:
  `venv\Scripts\python -m pytest tests\test_coding_agent_live_llm.py::test_hard_gate_real_github_character_image_response -q -s --tb=short -m live_llm`
  - Explicit route settings: PM route model
    `qwen3.6-34b-80l-fable-5-heretic`, PM thinking enabled; programmer route
    same model, programmer thinking disabled; synthesis reused PM route.
  - Result: 1 passed in 217.25s.
  - Trace:
    `test_artifacts/llm_traces/coding_agent_phase1_live_llm__hard_gate_character_image_response__20260620T235818097448Z.json`
  - Review artifact:
    `test_artifacts/llm_reviews/coding_agent_character_image_thinking_comparison_review.md`
- One non-live deterministic suite rerun used invalid placeholder model
  `test-model`; two tests that exercise the live PM path failed with provider
  `model_not_found`. This run is recorded as a verification setup failure and
  is not counted as passing evidence.
- Final non-review deterministic verification passed:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_pm_programmer.py tests\test_coding_agent_reading_acceptance.py -q`
  - Route settings used the actual local model
    `qwen3.6-34b-80l-fable-5-heretic`.
  - Result: 72 passed in 21.07s.
- Compile verification passed:
  `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
  - Result: exit code 0.
- Final static gate rerun:
  - Gate-specific repository/question grep in `src\kazusa_ai_chatbot\coding_agent`:
    no production matches; `rg` exit code 1.
  - Retired shared code-reading route token grep in `src tests docs README.md`:
    no matches; `rg` exit code 1.
  - Non-ignored repo-surface retired shared route/helper grep:
    no matches; `rg` exit code 1.
  - Non-ignored repo-surface unsupported worker/synthesizer route token grep:
    no matches; `rg` exit code 1.
  - Current code-reading PM/programmer route grep in `src tests docs README.md`:
    expected new route contract matches only.
  - Separate synthesizer route grep in `src tests docs README.md`:
    no matches; `rg` exit code 1.
  - Whole-repo/full-source behavior grep in `code_reading`: only allowed
    prohibition text in `prompts.py`, `README.md`, and `synthesizer.py`.
- Stage 4 no-subagent independent code review was authorized by the user on
  2026-06-21 after the parent agent stopped before code review as required.
  The review was executed by the parent agent without spawning subagents.
- Stage 4 review finding 1: two deterministic tests still leaked into the
  live PM/programmer path when placeholder route settings were used.
  - Reproducing command:
    `venv\Scripts\python -m pytest tests\test_coding_agent_interface.py::test_answer_code_question_reads_real_phase0_local_checkout tests\test_coding_agent_reading.py::test_answer_cap_is_enforced_when_public_run_succeeds -q --tb=short`
  - Result: failed with provider `model_not_found`, proving the tests were
    not isolated from live LLM calls.
  - Fix: patch the public `code_reading.run(...)` seam in the interface test
    while preserving the real Phase 0 local checkout path, and patch
    `agent.run_reading_supervisor` at the imported public-run target.
  - Rerun result: 2 passed in 1.09s.
- Stage 4 review finding 2: managed checkout failure results exposed raw
  exception text that could include local temp paths, raw Git stderr, or cache
  identifiers.
  - Fix: public Phase 0 failure messages for managed clone and managed raw
    download now use generic public-safe messages while preserving specific
    limitations.
  - Regression:
    `tests\test_coding_agent_fetching.py::test_run_sanitizes_managed_checkout_failure_result`
  - Rerun result: 1 passed in 0.59s.
- Stage 4 review finding 3: managed raw-download cleanup failures could escape
  the public code-fetching boundary as uncaught filesystem errors.
  - Fix: cleanup/write/move failures are converted into `ManagedDownloadError`,
    and secondary cleanup failures no longer mask the primary write/move
    failure.
  - Regression:
    `tests\test_coding_agent_fetching.py::test_run_sanitizes_managed_raw_download_cleanup_failure`
  - Rerun result: 1 passed in 0.59s.
- Stage 4 review finding 4: nested implementation paths containing a `tools`
  directory were classified as scripts, which could demote real implementation
  evidence.
  - Fix: only top-level `bin`, `scripts`, or `tools` directories are treated
    as script paths by directory name; nested source files keep their extension
    based implementation classification.
  - Regression: expanded
    `tests\test_coding_agent_reading.py::test_repository_intelligence_classifies_sources_and_symbols`
    with `src/orders/tools/runtime.py`.
  - Rerun result: 1 passed in 0.70s.
- Stage 4 review finding 5: managed clone temporary storage used a shared `t`
  path for every source, creating cross-source collision risk.
  - Fix: managed clone temporary roots now include the compact cache hash
    (`t<hash>`) matching the checkout/metadata/lock path scheme.
  - Regression: expanded
    `tests\test_coding_agent_fetching.py::test_managed_checkout_paths_use_compact_hash_storage`
    to assert distinct temp roots for distinct repositories.
  - Rerun result: 1 passed in 0.61s.
- Stage 4 review finding 6: the synthesis prompt still contained one
  source-shaped algorithm example (`PID`) even though the production rule was a
  generic anti-inference rule.
  - Fix: removed the source-shaped example and kept the generic requirement
    that algorithm, architecture, framework, or design-pattern names must be
    grounded in selected evidence.
  - Static verification: gate/source-shaped production grep over
    `src\kazusa_ai_chatbot\coding_agent`, `src\kazusa_ai_chatbot\config.py`,
    and `src\kazusa_ai_chatbot\llm_interface\route_report.py` had no matches;
    `rg` exit code 1.
- Stage 4 focused review regression set passed with placeholder route
  settings:
  `venv\Scripts\python -m pytest tests\test_coding_agent_interface.py::test_answer_code_question_reads_real_phase0_local_checkout tests\test_coding_agent_reading.py::test_answer_cap_is_enforced_when_public_run_succeeds tests\test_coding_agent_fetching.py::test_run_sanitizes_managed_checkout_failure_result tests\test_coding_agent_fetching.py::test_run_sanitizes_managed_raw_download_cleanup_failure tests\test_coding_agent_fetching.py::test_managed_checkout_paths_use_compact_hash_storage tests\test_coding_agent_reading.py::test_repository_intelligence_classifies_sources_and_symbols -q --tb=short`
  - Result: 6 passed in 1.11s.
- Stage 4 full deterministic coding-agent suite passed with placeholder route
  settings, proving non-live tests do not depend on a real local LLM:
  `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_coding_agent_reading.py tests\test_coding_agent_reading_acceptance.py tests\test_coding_agent_reading_pm_programmer.py -q`
  - Result: 74 passed in 2.79s.
- Stage 4 route/config verification passed with explicit split route settings:
  `venv\Scripts\python -m pytest tests\test_config.py tests\test_llm_interface_route_report.py tests\test_coding_agent_reading_pm_programmer.py -q`
  - Result: 71 passed in 5.26s.
  - A prior invocation without explicit PM/programmer route environment failed
    during config import with missing `CODING_AGENT_PM_LLM_BASE_URL`; this is
    recorded as a command environment failure and corrected by rerunning with
    the required route settings.
- Stage 4 compile verification passed:
  `venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot\coding_agent`
  - Result: exit code 0.
- Stage 4 static verification:
  - `git diff --check`
    - Result: no whitespace errors; Git emitted existing line-ending warnings.
  - Retired shared route token grep over `src tests docs README.md`:
    no matches; `rg` exit code 1.
  - Gate/source-shaped production grep over the coding-agent runtime surface:
    no matches; `rg` exit code 1.
  - Raw managed-checkout failure string grep found only regression-test
    strings outside the production coding-agent surface.
- Stage 4 review conclusion: no unresolved review blockers remain. The plan's
  six hard real LLM gates remain recorded in the Stage 3 review artifact, and
  Stage 4 fixes were scoped to generic failure-boundary hardening, generic path
  handling, generic source classification, deterministic test isolation, and
  removal of source-shaped prompt text.
