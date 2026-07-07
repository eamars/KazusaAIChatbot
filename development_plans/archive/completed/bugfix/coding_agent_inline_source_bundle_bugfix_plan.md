# coding agent inline source bundle bugfix plan

## Summary

- Goal: Fix background and direct coding-agent code-reading tasks where the
  user provides one or more inline code fragments instead of a GitHub/local
  source, by resolving inline fragments into a managed read-only source
  contract consumed by the existing `code_reading` path.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `llm-trace-debug`, `py-style`,
  `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for the `code_fetching` source contract;
  compatible for existing GitHub, raw GitHub, local checkout, background-work,
  accepted-task, and dialog ownership boundaries.
- Highest-risk areas: source-material contract shape, local LLM span
  extraction, exact inline-code preservation, multi-fragment cardinality,
  explicit-source precedence, public-safe metadata for non-GitHub sources, and
  preventing background/dialog layers from learning source parsing.
- Acceptance criteria: the captured Z3 production case resolves as a managed
  inline source bundle and reaches `code_reading`; existing explicit GitHub,
  raw GitHub, repo-hint, and local-checkout cases keep their current behavior;
  ambiguous or unsafe mixed-source cases produce actionable `needs_user_input`
  or `rejected` outcomes; all new inline-source real LLM tests, prior
  source-intake real LLM signoff tests, affected existing coding/background
  real LLM regression tests, deterministic tests, patched handoff tests, and
  replay verification pass with reviewed evidence.

## Context

The production failure evidence is stored under:

```text
test_artifacts/dialog_z3_failure_review.md
test_artifacts/dialog_z3_adjacent_chat_history.json
test_artifacts/dialog_z3_background_jobs_window.json
test_artifacts/dialog_z3_accepted_tasks_window.json
test_artifacts/dialog_z3_failure_trace.json
test_artifacts/dialog_z3_event_log_window.json
```

The user supplied inline Python/Z3 code in QQ message `1812203673`. The live
turn accepted delayed work, the background worker selected `coding_agent`, and
the coding-agent supervisor selected `code_reading`. The worker then failed in
`code_fetching` with:

```text
source_resolver:no_source_found
Provide a public GitHub repository, tree, blob, raw file, or owner/repo source.
```

The root cause is a source-contract gap. Inline source text can enter
`task_brief`, but `answer_code_question()` always asks `code_fetching.run()`
for a source location before `code_reading.run()`. Current source fetching
resolves GitHub, raw GitHub, repo hints, and explicit local-checkout hints. It
does not materialize inline code text into a read-only source scope.

The fix belongs inside `code_fetching`, because `coding_agent/README.md`
states that `code_fetching` is the only source-resolution owner, and
`background_work/README.md` states that the generic background-work router
chooses only the worker. L2d, accepted-task lifecycle, background-work routing,
dialog, adapters, and code-reading PM/programmer workers must not parse source
syntax or infer inline-code semantics.

## Reference Files Read For Draft

- `README.md`
- `docs/HOWTO.md`
- `development_plans/README.md`
- `src/kazusa_ai_chatbot/coding_agent/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md`
- `src/kazusa_ai_chatbot/coding_agent/code_reading/README.md`
- `src/kazusa_ai_chatbot/background_work/README.md`
- `src/kazusa_ai_chatbot/accepted_task/README.md`
- `src/kazusa_ai_chatbot/coding_agent/models.py`
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/models.py`
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py`

The `development-plan` skill references `plan_contract.md`,
`execution_gates.md`, and `cutover_policy.md`; those files were not present in
this checkout. This draft follows the local registry and existing active and
completed plan formats.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, changing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing source-intake prompt
  contracts, model-facing payloads, source resolver boundaries, coding-agent
  handoff, or background-worker integration.
- `no-prepost-user-input`: load before changing how user-visible task text is
  interpreted into source roles, inline-code spans, source precedence, or
  clarification outcomes.
- `debug-llm`: load before running or reviewing real LLM source-intake tests,
  replaying production-like failures, or writing human-readable LLM review
  artifacts.
- `llm-trace-debug`: load before retrieving or comparing protected production
  trace evidence for replay/signoff.
- `py-style`: load before editing Python production or test files, and read
  both positive and negative constraint reference files in full.
- `cjk-safety`: load before writing Python prompts or tests that contain CJK
  text, CJK punctuation, or production CJK dialog/task fixtures.
- `test-style-and-execution`: load before adding, changing, running, or
  interpreting deterministic, patched LLM, real LLM, or E2E tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, documentation, and plan edits.
- Check `git status --short` before edits and before final sign-off.
- Do not read `.env`.
- Preserve the current high-level ownership chain:

  ```text
  accepted_task / background_work
    -> coding_agent worker
    -> handle_background_coding_task()
    -> code_fetching source resolution/materialization
    -> code_reading evidence-backed answer
    -> background_work result-ready cognition
  ```

- Keep source interpretation inside the coding-agent source boundary. Do not
  move inline-code extraction into L2d, dialog, accepted-task lifecycle,
  generic background-work router, adapter code, reading PM, or programmer
  workers.
- Keep the generic background-work router route-only. It may select
  `coding_agent`; it must not infer source type, code language, inline spans,
  filenames, source precedence, or clarification wording.
- Keep `code_reading` read-only. It must continue to consume a successful
  source contract and return bounded evidence; it must not fetch, materialize,
  parse chat text, run code, install packages, or mutate files.
- LLM stages own semantic source-role extraction. Deterministic code owns
  anchoring, cardinality, size limits, filename/path safety, workspace
  materialization, supported-source execution, status mapping, and public-safe
  metadata.
- The source-intake LLM must not be the authority for copied code content. It
  may identify source spans and roles; deterministic code must extract exact
  text from the trusted task text.
- Explicit trusted source fields remain authoritative. If `source_url`,
  `repo_url`, `repo_hint`, `local_root_hint`, or `local_path_hint` is present
  and invalid, unsafe, inaccessible, or unsupported, return the typed outcome
  for that explicit field instead of silently replacing it with inline code
  from `question`.
- Do not add compatibility shims, alias modules, fallback mappers, feature
  flags, dual old/new source-resolution paths, generic web browsing, package
  registry lookups, paste-site fetchers, attachment/OCR support, command
  execution, code execution, repository mutation, or multi-repository reading
  in this plan.
- Do not add a new LLM route or environment variable. Source intake continues
  to use `CODING_AGENT_PM_LLM`.
- Preserve percent-encoded URLs and Unicode task text. Do not normalize,
  decode, or reserialize candidate source text to justify an LLM-invented span.
- Public responses and worker metadata must not expose local roots,
  workspace roots, cache keys, raw command output, full source files, `.env`
  content, secret-like file content, `.git` internals, binary content, or
  adapter/internal queue identifiers.
- Runtime prompts must stay short, role-specific, and contract-oriented.
  Stable contract material belongs in the system prompt; current task text and
  retry feedback belong in the human payload.
- Real LLM tests must run one case at a time, with durable raw evidence and an
  agent-authored readable review before quality claims.
- After automatic context compaction, the active agent must reread this entire
  plan before continuing implementation, verification, handoff, lifecycle
  updates, or final reporting.
- Before final completion, lifecycle changes, merge, or sign-off, run an
  independent code review gate and record findings in `Execution Evidence`.

## Must Do

- Add inline source bundles as a first-class supported source family inside
  `code_fetching`.
- Extend the canonical source contract so successful source fetching can
  return GitHub/local/raw sources or a managed inline bundle through one
  downstream shape.
- Materialize inline snippets as managed read-only source files under the
  configured coding workspace.
- Preserve exact inline code text from the original task text, including
  indentation, CJK, punctuation, symbols, and percent-encoded strings.
- Support one inline fragment, multiple inline fragments for one task, fenced
  code blocks, unfenced but anchored code fragments, inline diffs/patches, and
  filename/language hints when safely available.
- Treat stack traces, logs, and pasted error output as supporting task context,
  not primary source, unless the user explicitly asks to analyze the log itself
  as the source artifact.
- Preserve current explicit GitHub, raw GitHub, repo-hint, local checkout, and
  local path behavior.
- Keep direct `answer_code_question()` and background
  `handle_background_coding_task()` on the same source-fetching path.
- Update `coding_agent` and `code_fetching` ICDs and architecture diagrams to
  reflect inline source materialization.
- Add deterministic tests for inline materialization, exact text preservation,
  safe filenames, source precedence, status mapping, mixed-source ambiguity,
  size/count limits, public metadata, and no local-path leakage.
- Add patched LLM tests for source-intake output handling, retry feedback,
  background coding handoff, and failure-to-accepted-task mapping.
- Add ten positive and ten negative/failure-mode real LLM source-intake tests
  for inline code snippets, mixed sources, ambiguity, unsafe input, and
  unsupported input. Run and inspect these one at a time.
- Add a production replay or E2E-style test using the captured Z3 task text as
  the final sign-off case.
- Record verification evidence and human-readable LLM review artifacts under
  `test_artifacts/`.

## Deferred

- Do not implement code execution or Z3 solving.
- Do not run user code, install packages, invoke shells, create virtualenvs, or
  validate program output in this plan.
- Do not modify existing repository source or apply patches through the coding
  agent.
- Do not implement attachment OCR, image-code extraction, paste-site fetchers,
  Gist fetching, GitLab/Bitbucket providers, package registry lookup, archive
  extraction, PR/issue/discussion content fetching, or generic web browsing.
- Do not add multi-repository or multi-independent-source reading. Multiple
  inline fragments can form one managed bundle only when they are one task.
- Do not change accepted-task identity, background-work queue schema,
  dispatcher delivery, dialog wording prompts, cognition prompts, adapters,
  scheduler, or memory/consolidation behavior.
- Do not add broad LLM retry loops or repair agents. Keep the existing bounded
  one-retry source-intake pattern only for localized extraction failures.
- Do not make deterministic keyword classifiers decide whether raw user text
  is code.

## Cutover Policy

Overall strategy: bigbang inside `code_fetching`; compatible outside the
source boundary.

| Area | Policy | Instruction |
|---|---|---|
| Source contract | bigbang | Generalize the canonical source contract once so it can represent GitHub/local/raw and managed inline bundle sources. Do not add parallel response shapes or fallback adapters. |
| Inline question-text sources | bigbang | Resolve answerable inline code through source-intake plus deterministic managed materialization instead of returning generic `no_source_found`. |
| Explicit source fields | compatible | Preserve current explicit source precedence and failure behavior. Invalid explicit fields remain authoritative. |
| GitHub/local/raw source behavior | compatible | Preserve existing clone, raw download, local checkout, scope validation, and public status behavior except for type names needed by the generalized source contract. |
| `code_reading` consumer path | compatible | Keep `code_reading.run()` consuming a successful repository/source-scope pair and returning bounded evidence. |
| Background-work router | compatible | Keep route-only worker selection and existing result mapping shape. |
| Accepted-task lifecycle | compatible | Preserve accepted-task states, result/failure semantics, duplicate identity, and result-ready cognition handoff. |
| Public safety | bigbang | Ensure inline-bundle metadata never exposes local roots, workspace roots, cache keys, raw full source files, or private internals. |
| Tests | bigbang | Add focused inline-source tests and update existing tests that assert GitHub-only provider/storage assumptions. |

## Cutover Policy Enforcement

- For bigbang areas, update caller, callee, tests, docs, and ICDs in one
  canonical contract change.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Do not add compatibility branches merely to preserve old internal type
  assumptions such as `provider == "github"` where the canonical contract now
  needs `inline`.
- Any change to the cutover policy requires user approval before
  implementation.

## Target State

The target source-resolution path is:

```text
CodeFetchingRequest
  -> explicit trusted source fields and local hints, when present
  -> source-intake specialist for question text when no explicit source or
     local hint is present
  -> deterministic resolver
  -> one resolved source material:
       GitHub clone
       raw GitHub file download
       existing local checkout scope
       managed inline source bundle
  -> CodeRepositoryRef + CodeSourceScope
  -> code_reading.run()
```

Inline source materializes into a managed read-only directory or file:

```text
{CODING_AGENT_WORKSPACE_ROOT}/
  inline_sources/
    {stable_content_or_task_hash}/
      fragment_001.py
      fragment_002.txt
      manifest.json
```

Public metadata uses safe source labels:

```python
{
    "provider": "inline",
    "owner": "inline",
    "repo": "{public_safe_bundle_id}",
    "source_url": "inline://accepted-task/{public_safe_id}",
    "requested_ref": None,
    "resolved_ref": "inline",
    "current_commit": "inline-sha256:{hash}",
    "default_branch": "",
    "storage_kind": "managed_inline_bundle",
    "managed_checkout": True,
    "dirty_state": "clean",
}
```

Internal metadata may include the local bundle root and workspace root for
downstream reading only. Public responses, worker metadata, traces, and
accepted-task surfaces must not expose those internal paths.

## System Contract Audit

| Stage | Current owner | Plan instruction |
|---|---|---|
| Adapter and brain intake | Platform normalization and typed message envelope | No change. Do not parse inline code here. |
| L2d and accepted-task action | Decide whether delayed coding work is accepted | No change. Do not add code/source extraction here. |
| Accepted-task lifecycle | Persistence, duplicate identity, result state | No schema change. Existing task summary remains the prompt-visible request. |
| Background-work router | Select worker only | No source parsing. Route to `coding_agent` as today. |
| Coding-agent worker adapter | Inject workspace root and map sanitized results | No semantic parsing. Continue passing task text to `handle_background_coding_task()`. |
| Coding-agent supervisor | Select `code_reading`, `code_writing`, or `unsupported` | Keep read-vs-write ownership here. Do not make it parse inline spans. |
| `code_fetching` | Resolve source material into a safe local source contract | Add inline-bundle source family here. |
| `code_reading` | Evidence-backed read-only analysis of a successful source scope | Consume managed inline bundle like any other safe text source scope. |
| Dialog and L3 | Final visible wording | No prompt change. Better worker outcomes should supply actionable result/failure state. |

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Placement | Add inline source handling inside `code_fetching`. | This preserves the existing source owner and benefits direct and background callers. |
| Shape | Generalize `CodeRepositoryRef` and `CodingAgentRepositorySummary` rather than adding a parallel inline response. | `code_reading` already has a source-contract consumer; one canonical shape avoids branching readers. |
| Materialization | Store inline snippets as managed read-only files under `CODING_AGENT_WORKSPACE_ROOT`. | Existing readers operate on files/scopes and can reuse safe text evidence collection. |
| LLM output | Source-intake returns semantic roles and anchors/spans, not copied code as authority. | Local LLMs can corrupt code; deterministic slicing preserves exact task text. |
| Cardinality | Allow multiple inline fragments only as one bundle for one task. | This supports real chat usage without implementing multi-source reading. |
| Mixed sources | Ask clarification when multiple primary sources are present. | Silent selection can produce incorrect analysis. |
| Explicit fields | Keep explicit fields authoritative. | Trusted caller fields must not be overridden by untrusted question text. |
| Unsupported contexts | Keep image/OCR, paste sites, Gists, registries, archives, and non-GitHub providers out of scope. | The goal is inline source materialization, not new external provider support. |
| Retry | Retain one source-intake retry only for extraction-shaped failures. | Bounded recovery is enough for local LLM extraction mistakes. |
| Failure wording | Return actionable limitations from source fetching. | Result-ready cognition can render useful wording without dialog-specific fixes. |

## Contracts And Data Shapes

### Public Request Contract

Extend these request shapes with an optional trusted inline-source field:

```python
inline_sources: list[InlineSourceInput]
```

- `CodeFetchingRequest`
- `CodingAgentRequest`
- `CodingAgentBackgroundRequest`

`InlineSourceInput` should contain bounded source material already trusted by
the caller or deterministic extractor:

```python
{
    "content": str,
    "filename_hint": str | None,
    "language_hint": str | None,
    "source_label": str,
}
```

Question-text inline snippets must first pass through source-intake and
deterministic anchoring, then be converted into the same internal
`InlineSourceInput` shape before materialization. Direct callers may supply
`inline_sources` only when they already own trusted source text. This plan does
not add `inline_sources` to `CodingAgentWriteRequest`; existing-source writing
remains outside current scope.

Do not expose `InlineSourceInput` through background-work queue schema unless a
future deterministic action handler validates worker-specific payloads. The
normal background path should continue using `task_brief` and
`source_summary`.

### Source-Intake Output

Update the existing source-intake output to include inline families without
creating a new agent:

```python
{
    "task_source_mode": (
        "single_primary | inline_bundle | mixed_primary_with_context | "
        "compare_sources | source_free | unclear"
    ),
    "source_mentions": [
        {
            "raw_text": str,
            "role": (
                "primary_code_source | supporting_context | "
                "comparison_source | scope_modifier | reference_only | unknown"
            ),
            "family_hint": (
                "repository_url | repository_hint | local_path | raw_file | "
                "issue_or_pr | documentation_url | package_reference | "
                "archive_url | paste_or_gist | inline_code | inline_diff | "
                "log_or_trace | attachment | unknown_url | unknown"
            ),
            "language_hint": str,
            "filename_hint": str
        }
    ]
}
```

Normalizer rules:

- Unknown enum values normalize to `unclear` or `unknown`.
- Non-list `source_mentions` normalizes to an empty list.
- `raw_text`, `language_hint`, and `filename_hint` are bounded strings.
- `raw_text` must anchor to the original task text or to a deterministic
  visible span derived from that text.
- The prompt must instruct the LLM to extract only text visible in the task,
  preserve exact URL text and code anchors, avoid inventing filenames, and use
  supporting roles for stack traces/logs unless the log is the analysis target.

### Deterministic Resolver

The resolver must produce one internal outcome:

```text
succeeded: external source or inline bundle selected
needs_user_input: source exists but user correction or clarification is needed
rejected: source class is unsupported, unsafe, or out of policy
failed: internal preparation/materialization/access failure
```

New or updated internal issue codes:

```text
inline_source_resolved
inline_source_anchor_failed
inline_source_too_large
inline_source_too_many_fragments
inline_source_ambiguous_primary
inline_source_unsafe_content
inline_source_materialization_failed
mixed_primary_sources
supporting_context_only
image_only_source
```

Existing issue codes such as `no_source_found`, `unsupported_provider`,
`unsupported_source_family`, `malformed_source`, `source_not_visible_in_request`,
`unsupported_multi_source`, `path_not_found`, and `ref_not_found` remain valid.

### Managed Inline Bundle

Add an internal materializer owned by `code_fetching`, for example:

```text
src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_inline.py
```

Responsibilities:

- Create deterministic bundle directory names from content hash and source
  provenance.
- Write source fragments with safe generated filenames.
- Preserve optional safe filename hints after path traversal and reserved-name
  validation.
- Write a manifest with source labels, language hints, content hashes, and
  provenance.
- Return a `CodeRepositoryRef` and `CodeSourceScope` compatible with
  `code_reading`.
- Keep local paths internal.

`manifest.json` is diagnostic/internal. Public responses should not include
full source content or local file paths from the manifest.

## Source Use Cases

The implementation must include deterministic or real LLM coverage for these
cases:

1. One fenced Python code block and a request to review correctness.
2. One unfenced but anchored code fragment in chat text.
3. Multiple fenced fragments that form one task.
4. Multiple fragments with safe filename hints.
5. Inline diff/patch text asking for review.
6. Inline code plus stack trace as supporting context.
7. Inline code plus natural-language requirements.
8. GitHub repo plus pasted error log.
9. GitHub repo plus inline snippet where the repo is primary.
10. GitHub repo plus inline snippet where both are primary and ambiguous.
11. Explicit invalid `repo_url` plus valid inline code in `question`.
12. Unsupported URL plus valid inline code marked reference-only.
13. Image-only code mention.
14. Truncated code block.
15. Oversized inline snippet.
16. Too many inline fragments.
17. Secret-like pasted content.
18. Non-code prose mentioning code but no actual source.
19. Inline shell command or command output that should be supporting context.
20. Captured production Z3 accepted-task text.

## Prior Real LLM Baseline From Last Iteration

The prior source-intake iteration produced these real LLM tests and signoff
cases. They are part of the regression gate for this plan because this plan
extends the same source-intake and deterministic resolver boundary.

Focused source-intake live tests:

- `tests/test_coding_agent_source_intake.py::test_live_source_intake_extracts_captured_github_task`
- `tests/test_coding_agent_source_intake.py::test_live_source_intake_marks_unsupported_web_url`
- `tests/test_coding_agent_source_intake.py::test_live_source_intake_marks_multiple_repos_mode`

Twenty-case source-intake signoff harness:

- Fixture: `tests/fixtures/coding_agent_source_intake_signoff_cases.json`
- Harness:
  `tests/test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_case_live_llm`
- Case ids: `csi_001_github_trailing_cjk_prose`,
  `csi_002_markdown_repo_link`, `csi_003_github_tree_directory_scope`,
  `csi_004_github_blob_file_anchor`, `csi_005_raw_github_file`,
  `csi_006_owner_repo_hint`, `csi_007_percent_encoded_non_github_url`,
  `csi_008_unsupported_hosts_no_probe`,
  `csi_009_explicit_package_reference`,
  `csi_010_bare_package_word_not_source`,
  `csi_011_github_issue_target_content`,
  `csi_012_github_issue_ancillary_repo_analysis`,
  `csi_013_multi_repo_compare_unsupported`,
  `csi_014_multi_repo_ambiguous_primary`,
  `csi_015_same_repo_nested_scope`,
  `csi_016_same_repo_conflicting_files`,
  `csi_017_required_supporting_docs_unsupported`,
  `csi_018_optional_supporting_docs_limitation`,
  `csi_019_raw_local_path_in_chat`,
  `csi_020_explicit_invalid_source_authoritative`.

Affected existing coding/background real LLM regression files:

- `tests/test_coding_agent_live_llm.py`
- `tests/test_coding_agent_pm_lifecycle_role_live_llm.py`
- `tests/test_coding_agent_phase2_new_artifact_role_live_llm.py`
- `tests/test_coding_agent_phase2_new_artifact_e2e_live_llm.py`
- `tests/test_coding_agent_phase3_live_e2e.py`
- `tests/test_background_work_router_live_llm.py`
- `tests/test_background_work_text_artifact_live_llm.py`
- `tests/test_background_work_future_speak_live_llm.py`

The repository contains broader non-coding live LLM suites. They are outside
this plan's affected boundary unless the user explicitly expands this plan's
scope.

## Required New Inline-Source Real LLM Matrix

Replace the two placeholder inline live checks with twenty named real LLM
cases: ten positive cases and ten negative/failure-mode cases. These cases are
additional to the prior source-intake baseline above.

| Polarity | Test node id | Required evidence |
|---|---|---|
| Positive | `test_inline_source_single_fenced_python_live_llm` | One fenced Python block is selected as primary inline code and materializes as one file. |
| Positive | `test_inline_source_unfenced_python_anchor_live_llm` | Unfenced but clearly anchored code is selected without losing indentation or symbols. |
| Positive | `test_inline_source_multiple_fragments_one_task_live_llm` | Multiple fragments for one task become one inline bundle. |
| Positive | `test_inline_source_filename_hint_live_llm` | Safe filename hint is preserved; content remains exact. |
| Positive | `test_inline_source_diff_review_live_llm` | Inline diff is treated as primary source material for review. |
| Positive | `test_inline_source_code_plus_stack_trace_live_llm` | Code is primary; stack trace is supporting context. |
| Positive | `test_inline_source_code_plus_requirements_live_llm` | Code is primary; requirements prose remains task context. |
| Positive | `test_inline_source_cjk_prompt_python_code_live_llm` | CJK task text and punctuation do not corrupt code anchors. |
| Positive | `test_inline_source_markdown_language_fence_live_llm` | Markdown language fence informs language hint without becoming authority for copied code. |
| Positive | `test_inline_source_production_z3_replay_live_llm` | Captured production Z3 snippet resolves as an inline bundle and reaches the reading path. |
| Negative | `test_inline_source_no_source_asks_for_source_live_llm` | Code-reading request with no source returns `needs_user_input`. |
| Negative | `test_inline_source_image_only_needs_text_live_llm` | Image-only source mention asks for text source; no inline bundle is created. |
| Negative | `test_inline_source_truncated_code_needs_resend_live_llm` | Truncated snippet asks for a complete smaller source. |
| Negative | `test_inline_source_oversized_needs_narrowing_live_llm` | Oversized inline source returns a typed size outcome. |
| Negative | `test_inline_source_too_many_fragments_needs_narrowing_live_llm` | Too many fragments asks the user to narrow scope. |
| Negative | `test_inline_source_secret_like_content_needs_redaction_live_llm` | Secret-like pasted content asks for redaction and is not materialized. |
| Negative | `test_inline_source_mixed_primary_sources_needs_clarification_live_llm` | GitHub plus inline code as competing primaries asks for clarification. |
| Negative | `test_inline_source_explicit_invalid_repo_stays_authoritative_live_llm` | Invalid explicit trusted source field fails explicitly and does not fall back to inline code. |
| Negative | `test_inline_source_unsupported_reference_only_does_not_block_inline_live_llm` | Unsupported reference-only URL does not become primary or cause unsupported-source failure when inline code is the task source. |
| Negative | `test_inline_source_log_only_supporting_context_no_primary_live_llm` | Log-only supporting context does not become code source unless the user asks to analyze the log itself. |

Each new real LLM case must write a trace artifact under
`test_artifacts/llm_traces/coding_agent_inline_source_live_llm/` that records
case id, task text, raw source-intake output, normalized mentions,
deterministic anchors, resolver outcome, materialization outcome when present,
public-safe result metadata, and human inspection notes.

Hard gates for every new inline-source real LLM case:

- The real LLM output parses through the normal source-intake normalizer.
- Any selected source text is visible in the original task text or trusted
  request fields.
- Deterministic code, not the LLM, extracts exact inline source content.
- Positive cases reach `succeeded` and materialize the expected inline bundle.
- Negative cases reach the required typed `needs_user_input`, `rejected`, or
  `failed` outcome and do not silently choose a different source.
- Public-safe metadata contains no local roots, workspace roots, cache keys,
  raw command output, `.env`, `.git`, adapter ids, or internal queue ids.

## Failure Mode Matrix

| Failure mode | Required conclusion |
|---|---|
| No source in a code-reading request | `needs_user_input`: ask for repo, link, or code text. |
| Valid inline source anchors cleanly | `succeeded`: managed inline bundle. |
| LLM detects inline source but anchor fails | Retry source intake once; then `needs_user_input`. |
| LLM emits code not visible in task text | Reject extraction, retry once, then `needs_user_input`. |
| Multiple inline fragments for one task | `succeeded`: one managed inline bundle. |
| Multiple independent primary sources | `needs_user_input`: ask which source to inspect. |
| GitHub source plus supporting log | `succeeded`: GitHub source, log remains task context. |
| Explicit invalid source field plus inline source | Return explicit-source failure; do not fall back to inline. |
| Inline source exceeds size limit | `needs_user_input`: ask for a repo, file, or smaller scope. |
| Too many fragments | `needs_user_input`: ask to narrow the task or provide a repo. |
| Unsafe filename hint | Use generated safe filename and continue. |
| Secret-like source content | `needs_user_input`: ask for redacted code. |
| Image-only code | `needs_user_input`: ask for text source. |
| Unsupported external provider only | Existing unsupported-source response. |
| Workspace write/materialization failure | `failed`: public-safe internal preparation failure. |

## Implementation Checklist

### 1. Contract And Documentation

- Update `src/kazusa_ai_chatbot/coding_agent/code_fetching/models.py` so
  source metadata can represent `provider="inline"` and
  `storage_kind="managed_inline_bundle"`.
- Update `src/kazusa_ai_chatbot/coding_agent/models.py` to mirror public-safe
  repository summary changes.
- Update `src/kazusa_ai_chatbot/coding_agent/README.md` and
  `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md`.
- Keep background-work and accepted-task ICDs unchanged unless implementation
  reveals a documentation-only clarification is needed.

### 2. Source-Intake Prompt And Parser

- Update `source_intake.py` prompt to identify inline code, inline diff, and
  log/supporting context roles.
- Keep the prompt plain and role-specific; avoid implementation history and
  test-shaped examples.
- Preserve existing GitHub/source mention behavior.
- Add normalizer handling for new source family hints and language/filename
  hints.

### 3. Deterministic Resolver

- Extend `source_resolver.py` to select an inline bundle when the LLM marks one
  or more anchored inline fragments as the primary source for one task.
- Preserve explicit-source precedence.
- Add mixed-source clarification rules.
- Add size, count, and safety checks.
- Keep one bounded retry only for extraction-shaped failures.

### 4. Managed Inline Materializer

- Add `managed_inline.py` or equivalent inside `code_fetching`.
- Materialize exact extracted source text into safe files under
  `CODING_AGENT_WORKSPACE_ROOT` or the request workspace root.
- Return generalized `CodeRepositoryRef` and `CodeSourceScope`.
- Ensure source scope points at either one file or the bundle directory.

### 5. Fetching Orchestration

- Update `code_fetching.agent.run()` to dispatch selected inline sources to the
  managed inline materializer.
- Preserve managed clone/download/local checkout branches.
- Ensure public limitations and trace summaries are source-specific and safe.

### 6. Consumer Compatibility

- Update `_repository_summary()` and public sanitizers so inline metadata is
  public-safe.
- Verify `answer_code_question()` passes inline-backed source contracts to
  `code_reading.run()` without special reader branches.
- Verify background coding worker result mapping handles inline provider and
  source scope without leaking local paths.

### 7. Observability

- Add trace summaries such as:

  ```text
  source_resolver:inline_bundle_resolved
  managed_inline:materialized:{fragment_count}
  ```

- Keep trace summaries public-safe and bounded.
- Do not store full source text in worker metadata.

### 8. Verification And Review

- Add and run deterministic tests.
- Add and run patched handoff tests.
- Add and run real LLM source-intake tests one at a time.
- Run the captured Z3 replay/E2E sign-off.
- Run focused existing coding-agent regression tests.
- Run independent code review gate and remediate findings.
- Update this plan's execution evidence before completion.

## Verification Commands

Use focused commands during implementation. Add the planned inline-source
cases to the existing coding-agent source-intake, source-resolution, fetching,
interface, and background-work tests, plus one real LLM inline-source test file
named `tests\test_coding_agent_inline_source_live_llm.py`. Preserve the test
categories and one-at-a-time real LLM rule.

Deterministic and patched tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake.py -q
venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_resolution.py -q
venv\Scripts\python.exe -m pytest tests\test_coding_agent_fetching.py -q
venv\Scripts\python.exe -m pytest tests\test_coding_agent_interface.py -q
venv\Scripts\python.exe -m pytest tests\test_background_work_coding_agent.py -q
```

Production replay or E2E sign-off:

```powershell
venv\Scripts\python.exe -m pytest tests\test_coding_agent_phase3_handoff_e2e.py -q
```

Real LLM tests must run individually with `-q -s`, and each output must be
inspected before running the next case. Do not batch real LLM cases.

Prior focused source-intake live regression cases:

```powershell
venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake.py::test_live_source_intake_extracts_captured_github_task -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake.py::test_live_source_intake_marks_unsupported_web_url -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake.py::test_live_source_intake_marks_multiple_repos_mode -q -s -m live_llm
```

Prior 20-case source-intake signoff harness:

```powershell
$env:CODING_AGENT_SOURCE_INTAKE_SIGNOFF_CASE_ID = "csi_001_github_trailing_cjk_prose"; venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_case_live_llm -q -s -m live_llm
```

Repeat the harness command one case at a time for every prior signoff case id
listed in `Prior Real LLM Baseline From Last Iteration`, inspecting the trace
artifact and behavior before moving to the next case.

New inline-source real LLM cases:

```powershell
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_single_fenced_python_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_unfenced_python_anchor_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_multiple_fragments_one_task_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_filename_hint_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_diff_review_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_code_plus_stack_trace_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_code_plus_requirements_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_cjk_prompt_python_code_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_markdown_language_fence_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_production_z3_replay_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_no_source_asks_for_source_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_image_only_needs_text_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_truncated_code_needs_resend_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_oversized_needs_narrowing_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_too_many_fragments_needs_narrowing_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_secret_like_content_needs_redaction_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_mixed_primary_sources_needs_clarification_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_explicit_invalid_repo_stays_authoritative_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_unsupported_reference_only_does_not_block_inline_live_llm -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests\test_coding_agent_inline_source_live_llm.py::test_inline_source_log_only_supporting_context_no_primary_live_llm -q -s -m live_llm
```

Affected existing coding/background real LLM regression tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_coding_agent_live_llm.py tests\test_coding_agent_pm_lifecycle_role_live_llm.py tests\test_coding_agent_phase2_new_artifact_role_live_llm.py tests\test_coding_agent_phase2_new_artifact_e2e_live_llm.py tests\test_coding_agent_phase3_live_e2e.py tests\test_background_work_router_live_llm.py tests\test_background_work_text_artifact_live_llm.py tests\test_background_work_future_speak_live_llm.py --collect-only -q -m live_llm
```

Run every collected node id from the affected existing regression files one at
a time with `-q -s -m live_llm`. If a collected test also requires `live_db`,
run it only with the required database available and record that prerequisite
in `Execution Evidence`.

Suggested focused regression batch after the change:

```powershell
venv\Scripts\python.exe -m pytest tests\test_coding_agent_fetching.py tests\test_coding_agent_source_intake.py tests\test_coding_agent_source_resolution.py tests\test_coding_agent_interface.py tests\test_background_work_coding_agent.py -q
```

## Acceptance Criteria

- The captured Z3 task text resolves into a managed inline source bundle and
  reaches `code_reading` with evidence rows or an evidence-backed reading
  limitation.
- `source_resolver:no_source_found` is no longer produced for answerable
  inline code-review tasks.
- Existing explicit GitHub, raw GitHub, repo-hint, local-root, and local-path
  cases preserve current success and failure behavior.
- Explicit invalid source fields do not silently fall back to inline snippets.
- Multiple inline fragments for one task materialize as one bundle.
- Ambiguous mixed primary sources ask for clarification.
- Oversized, too-many-fragment, image-only, secret-like, and unsupported
  source cases return the required typed outcomes.
- Public response and worker metadata include no local roots, workspace roots,
  cache keys, raw command output, `.env`, `.git`, full source files, or
  adapter/internal queue ids.
- The ten positive and ten negative/failure-mode inline-source real LLM cases
  pass one at a time with trace artifacts and human inspection notes.
- The prior three focused source-intake live tests and prior 20-case
  source-intake signoff harness pass one at a time.
- Affected existing coding/background real LLM regression tests pass one node
  id at a time, including live DB prerequisites where marked.
- Documentation and architecture diagrams match the implemented source
  boundary.
- Independent code review finds no blocking issues, or all blocking issues are
  remediated and re-reviewed.

## Plan Review Findings And Remediation

Review focus: use cases, failure modes, real LLM pass criteria, consistency
with the prior source-intake plan, and risk of overfitting to the Z3 incident.

| Finding | Risk | Remediation in this plan |
|---|---|---|
| The prior draft said "representative real LLM tests" but did not identify the last iteration's real LLM baseline. | Execution could skip the existing source-intake regression cases and still appear complete. | Added `Prior Real LLM Baseline From Last Iteration` with the three focused live tests, 20 signoff case ids, and affected coding/background live regression files. |
| The prior draft only showed two placeholder inline live commands. | The new inline path would be under-tested against local LLM extraction variance. | Replaced placeholders with twenty named inline live cases: ten positive and ten negative/failure-mode cases. |
| Positive and negative inline behavior was not separated. | A single "passes" result could hide silent fallback, invented source spans, or unsupported-source misclassification. | Added hard gates for exact visibility, deterministic slicing, typed negative outcomes, and no silent alternate-source selection. |
| Existing real LLM regression scope was ambiguous. | The plan could either under-test affected coding/background paths or overreach into unrelated global live LLM suites. | Defined affected regression scope as coding-agent and background-work live LLM files touched by this source boundary, while noting broader live LLM suites are outside scope unless explicitly added. |
| The source-intake signoff harness was not connected to the new pass criteria. | The plan could pass new inline tests while regressing GitHub, percent-encoded URL, unsupported-provider, mixed-source, and explicit-source behavior. | Required prior focused source-intake tests and all 20 prior signoff cases to pass one at a time before completion. |
| Live DB prerequisites were not called out for background regression tests. | A live DB-marked regression could be skipped or misreported as a source-intake success. | Added a verification rule to collect affected live nodes and record live DB prerequisites in `Execution Evidence`. |

## Execution Evidence

- 2026-07-06: User explicitly approved execution without subagents. The active
  execution agent switched the plan status to `in_progress`; production-code
  changes are authorized only within this plan's stated change surface.
- 2026-07-06: Implemented managed inline source bundles inside
  `code_fetching`, including source-intake prompt/normalizer updates,
  deterministic inline anchoring and validation, managed materialization,
  public-safe inline repository metadata, direct caller `inline_sources`, and
  updated coding-agent/code-fetching ICD documentation.
- 2026-07-06: Deterministic and patched verification passed:
  `venv\Scripts\python.exe -m py_compile ...` on touched production/test
  Python files; focused suite
  `tests\test_coding_agent_source_intake.py`,
  `tests\test_coding_agent_source_resolution.py`,
  `tests\test_coding_agent_fetching.py`,
  `tests\test_coding_agent_interface.py`,
  `tests\test_background_work_coding_agent.py`,
  `tests\test_coding_agent_phase3_handoff_e2e.py`, and
  `tests\test_coding_agent_source_intake_signoff.py -q -m "not live_llm"`
  passed with `84 passed, 4 deselected`.
- 2026-07-06: New inline-source real LLM matrix passed 20/20 cases one at a
  time. The captured production-like Z3 replay
  `test_inline_source_production_z3_replay_live_llm` resolved as
  `inline_code:primary_code_source`, status `succeeded`, and materialized one
  managed inline fragment.
- 2026-07-06: Prior focused source-intake live regressions passed one at a
  time: captured GitHub task, unsupported web URL, and multiple-repos mode.
- 2026-07-06: Prior 20-case source-intake signoff harness passed one case at a
  time under the final retry-aware harness. Notable inspected cases:
  `csi_014_multi_repo_ambiguous_primary` corrected from first-pass
  `compare_sources` to final `unclear` and
  `needs_user_input/ambiguous_primary_source`; `csi_017_required_supporting_docs_unsupported`
  kept documentation URL as `supporting_context` and returned
  `needs_user_input/required_supporting_source_unsupported`; `csi_020_explicit_invalid_source_authoritative`
  preserved explicit malformed source precedence.
- 2026-07-06: Human-readable LLM review artifact written to
  `test_artifacts/llm_reviews/coding_agent_inline_source_bundle_review_20260706.md`.
- 2026-07-06: Affected existing live regression execution was attempted.
  `tests\test_coding_agent_live_llm.py::test_hard_gate_real_github_character_image_response`
  passed and wrote a successful trace. The next hard-gate node
  `test_hard_gate_real_github_home_assistant_entity_action` timed out after
  15 minutes with no trace. `test_live_pm_handles_ambiguous_or_too_broad_request`
  failed because the live PM returned `need_programmers` instead of the legacy
  expected `needs_user_input` or `overloaded`. The background router live case
  passed. `test_background_work_text_artifact_live_case` failed because the
  generator returned a Markdown code fence rather than the JSON artifact
  contract. These failures did not exercise the new inline source
  resolver/materializer, but they block full plan acceptance under the current
  acceptance criteria.
- 2026-07-07: Final focused deterministic verification passed before full
  non-live regression:
  `venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake.py tests\test_coding_agent_source_resolution.py tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_background_work_coding_agent.py tests\test_coding_agent_phase3_handoff_e2e.py tests\test_coding_agent_source_intake_signoff.py -q -m "not live_llm"`
  completed with `84 passed, 4 deselected`.
- 2026-07-07: The user explicitly deferred the remaining whole-project
  summary regression as a separate architecture issue outside this closure.
  The two stale PM lifecycle fixture failures were corrected and verified with
  exact one-at-a-time live LLM reruns:
  `tests\test_coding_agent_pm_lifecycle_role_live_llm.py::test_live_pm_gate_02_source_report_tests_programmer_task`
  and
  `tests\test_coding_agent_pm_lifecycle_role_live_llm.py::test_live_pm_gate_04_tests_repeat_source_literals`.
  Fresh traces:
  `test_artifacts\llm_traces\coding_agent_pm_lifecycle_role_live_llm__gate_02_source_report_tests_programmer_task__20260707T015905908309Z.json`
  and
  `test_artifacts\llm_traces\coding_agent_pm_lifecycle_role_live_llm__gate_04_tests_repeat_source_literals__20260707T015939562561Z.json`.
  Before/after trace comparison confirmed the root cause was incomplete
  producer facts in positive live fixtures, not a PM prompt defect.
- 2026-07-07: Full non-live suite was run with
  `venv\Scripts\python.exe -m pytest -q`. It completed with `2871 passed`,
  `2 skipped`, `524 deselected`, and `6 failed`. The branch-related coding
  failures were traced to search-evidence overlap pruning and synthesis
  diagnostic limitation filtering; both were fixed during final cleanup.
- 2026-07-07: Post-cleanup focused branch gate passed:
  `venv\Scripts\python.exe -m pytest tests\test_coding_agent_source_intake.py tests\test_coding_agent_source_resolution.py tests\test_coding_agent_fetching.py tests\test_coding_agent_interface.py tests\test_background_work_coding_agent.py tests\test_coding_agent_phase3_handoff_e2e.py tests\test_coding_agent_source_intake_signoff.py tests\test_coding_agent_reading.py::test_search_evidence_keeps_more_than_three_source_regions tests\test_coding_agent_reading_acceptance.py::test_public_run_keeps_grounded_prose_when_synthesis_omits_answer_text -q -m "not live_llm"`
  completed with `86 passed, 4 deselected`.
- 2026-07-07: Three non-live full-suite failures remained after branch cleanup
  and were confirmed outside the inline-source branch diff relative to `main`:
  `tests\test_control_console_config_routes.py::test_brain_model_route_api_applies_and_resets_selected_route`
  expects 13 brain model routes while the runtime reports 14;
  `tests\test_multi_source_cognition_stage_07_reflection_dry_run.py::test_text_chat_prompt_fingerprints_remain_stable`
  and
  `tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py::test_existing_l1_l2_l3_prompt_bytes_are_unchanged`
  expect `_COGNITION_SUBCONSCIOUS_PROMPT` length 3395 while the current prompt
  length is 3335. These baseline failures block local merge under the final
  branch-completion rule and should be handled in a separate cleanup.

## Independent Code Review

Required before completion. The reviewer must focus on:

- source ownership boundaries;
- explicit-source precedence;
- public metadata leakage;
- inline text preservation;
- local path safety;
- local LLM prompt fragility;
- test coverage for the failure matrix;
- absence of dialog/background-work routing workarounds.

2026-07-06 manual no-subagent review: no blocking issue found in the inline
source ownership boundary, managed inline path safety, public metadata
sanitization, explicit-source precedence, or inline exact-text preservation.
Residual risk remains in local LLM role stability for optional versus required
supporting sources and compare versus ambiguous multi-source wording; this is
covered by bounded retry feedback plus real LLM signoff traces listed above.

## Completion And Lifecycle

Completed on 2026-07-07 after final focused deterministic verification,
targeted live LLM reruns, user-approved deferral of the remaining architecture
issue, and registry/archive cleanup.
