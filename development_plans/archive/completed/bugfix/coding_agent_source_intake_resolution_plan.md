# coding agent source intake resolution plan

## Summary

- Goal: Add a bounded source-intake specialist inside `code_fetching` so coding tasks with messy, mixed, missing, or unsupported source mentions resolve into one safe code source or a typed failure.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `cjk-safety`, `development-plan`.
- Overall cutover strategy: bigbang inside `code_fetching`; public request and response contracts stay stable.
- Highest-risk areas: LLM source-role extraction, mixed-source cardinality, unsupported-but-recognized source families, typed failure mapping, and avoiding generic web browsing or multi-source overreach.
- Acceptance criteria: captured production-style GitHub task text resolves through the background coding path, supported explicit sources keep existing behavior, mixed and unsupported sources return specific outcomes, and the 20-case final sign-off matrix is run one case at a time and inspected.

## Context

The current `code_fetching` package resolves explicit GitHub and local-checkout inputs, plus public GitHub URLs embedded in the raw `question` text. Its current embedded URL path is deterministic and late: `agent._select_github_source()` calls `github.extract_http_urls(question)`, then `parse_github_source()`, then `source_scope.choose_source()`.

That path failed for a production background coding task because a valid GitHub repository URL was followed by natural-language CJK punctuation and prose. The deterministic URL extractor swallowed the prose into the URL candidate, GitHub parsing rejected it, no supported source remained, and the worker returned generic `needs_user_input`.

The bug is one symptom of a wider missing boundary. `code_fetching` needs a source-intake step that separates source mention extraction from deterministic source resolution. The system must support broader user input shapes: clean URLs, markdown links, trailing prose, wrong URLs, issue/PR URLs, docs, package references, local hints, inline snippets, archives, multiple repositories, and mixed code/supporting references.

Current source contracts and ownership:

- `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md` owns source resolution.
- `src/kazusa_ai_chatbot/coding_agent/README.md` states that `code_fetching` is the only source-resolution owner.
- `src/kazusa_ai_chatbot/background_work/README.md` states that the background-work router chooses the worker only. Worker-local argument extraction belongs inside the worker or subagent.
- `CodingAgentRequest`, `CodingAgentWriteRequest`, and `CodingAgentBackgroundRequest` already expose `source_url`, `repo_url`, `repo_hint`, `local_root_hint`, `local_path_hint`, `requested_ref`, and `source_scope_hint`.

The new design keeps source intake inside `code_fetching`, not in the generic background router, not in L2d, and not in the reading PM.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, or changing this plan.
- `local-llm-architecture`: load before editing the source-intake prompt, LLM call, code-fetching handoff, or background coding flow.
- `no-prepost-user-input`: load before changing how raw user task text is interpreted into source roles.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before writing Python tests or prompts containing CJK punctuation or CJK source text.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.
- Preserve the public `CodeFetchingRequest` and `CodeFetchingResult` shapes.
- Keep source-intake output transient and internal to `code_fetching`.
- LLM stages own semantic extraction of source mentions and roles.
- Deterministic code owns URL grammar, canonicalization, provider support, local path safety, access checks, size limits, failure classification, and final source selection.
- Explicit trusted request fields are authoritative. If an explicit source field is malformed, unsafe, inaccessible, or unsupported, return the typed outcome for that field instead of silently replacing it with a question-text source.
- Deterministic `visible_source_spans` are observation aids only. They must not create source roles, promote a source candidate, or override source-intake semantics.
- Every LLM-proposed `raw_text` candidate must be anchored to the original task text or to a deterministic visible span derived from that text. Unanchored candidates are rejected as extraction failures.
- Do not move source extraction into `BACKGROUND_WORK_LLM`, L2d, the background-work router, the reading PM, or programmer workers.
- Do not add a new LLM route or environment variable. Use `CODING_AGENT_PM_LLM` for the source-intake specialist.
- Do not parse arbitrary local filesystem paths from untrusted user text into `local_root_hint` or `local_path_hint`. Local hints remain explicit trusted request fields.
- Do not probe unsupported HTTP hosts, URL shorteners, generic document sites, paste sites, package registries, or private-network URLs. Access checks run only for supported providers and explicit trusted local fields.
- Do not implement generic web browsing, package registry lookup, archive extraction, multi-repository reading, PR diff fetching, attachment processing, command execution, or repository mutation in this plan.
- If Python test files include CJK text, use single-quoted Python literals for those strings and run a Python syntax check after edits.

## Must Do

- Add a specialized source-intake LLM stage inside `code_fetching`.
- Add deterministic source-resolution logic that consumes explicit request fields and source-intake proposals through one canonical internal pipeline.
- Recognize broad source families and return typed outcomes for unsupported or ambiguous cases.
- Preserve current explicit GitHub, raw GitHub, `owner/repo`, local checkout, and local path behavior.
- Fix the captured production-style failure where a GitHub repo URL has trailing natural-language punctuation or prose.
- Add one bounded retry from source-intake only when deterministic validation says the failure is extraction-shaped.
- Add deterministic tests for source family recognition, cardinality, provider-aware canonicalization, unsupported inputs, local path safety, and retry gating.
- Add patched LLM handoff tests for source-intake output, resolver feedback, and background code-reading integration.
- Add real LLM source-intake tests and run them one at a time with inspected logs.
- Add the 20-case source-intake/resolver final sign-off matrix and require it before completion.
- Update `code_fetching` and `coding_agent` README diagrams or text to reflect the implemented source-intake stage.

## Deferred

- Do not add durable MongoDB fields, accepted-task schema fields, background-work queue fields, or new public source payloads.
- Do not add a new source provider fetcher beyond the currently supported GitHub/local fetchers.
- Do not add multi-repository code-reading execution.
- Do not fetch GitHub issue, PR, discussion, release, gist, or package registry content.
- Do not hand unsupported web/document URLs to `web_agent3` from `code_fetching`.
- Do not add broad retry loops, repair agents, compatibility shims, aliases, fallback mappers, or dual old/new source-selection paths.
- Do not change final dialog wording, accepted-task lifecycle, background-work routing, or adapter delivery.
- Do not change code-writing semantics for source-free new-artifact proposals.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Question-text source selection | bigbang | Replace direct `extract_http_urls(question)` source selection with source-intake plus deterministic resolver. |
| Explicit request fields | compatible | Preserve existing `source_url`, `repo_url`, `repo_hint`, `local_root_hint`, `local_path_hint`, `requested_ref`, and `source_scope_hint` behavior. |
| Public request/response shape | compatible | Keep existing typed request and result contracts. |
| Unsupported source families | bigbang | Recognize and classify unsupported families through one resolver path instead of silently falling through to generic no-source behavior. |
| Tests | bigbang | Add replacement and regression tests for the new source-intake path; update old tests only where their assumptions are replaced by the new contract. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `compatible`, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The target source-resolution path is:

```text
CodeFetchingRequest
  -> explicit trusted source fields and local hints, when present
  -> source-intake specialist for raw question text, when needed
  -> deterministic source resolver
  -> one resolved supported source or one typed terminal outcome
  -> existing managed clone, managed raw download, or local checkout resolution
```

The implemented architecture becomes:

```text
code_fetching
  source_intake [LLM, CODING_AGENT_PM_LLM]
    owns semantic source mention extraction and source roles
  source_resolver [deterministic]
    owns provider grammar, canonicalization, support, safety, access class,
    cardinality, retry feedback, and final selection
  existing GitHub/local fetchers [deterministic]
    own clone/download/local checkout mechanics
```

Normal call count:

```text
explicit source fields or local hints: 0 new LLM calls
question-only source mentions: 1 source-intake LLM call
repairable extraction failure: at most 1 additional source-intake LLM call
```

Final supported execution remains one code source per `CodeFetchingResult`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Placement | Put source intake inside `code_fetching`. | `code_fetching` is the source-resolution owner and is shared by direct and background coding paths. |
| LLM route | Use `CODING_AGENT_PM_LLM` with a separate source-intake prompt. | PM route owns coding semantic decomposition; a separate prompt prevents mixing extraction with reading/writing PM responsibilities. |
| Public schema | Keep existing request and result shapes. | Existing callers already have fields for explicit source data; new output can fit existing `limitations` and `trace_summary`. |
| Internal contract | Use one transient source-mention proposal shape. | One generic proposal contract covers all source families without a new data structure per incident. |
| Source family scope | Recognize broad families, execute only supported families. | Users may paste anything; recognizing unsupported families gives precise feedback without adding unsupported fetchers. |
| Mixed sources | Select one primary code source only. | Current reading/fetching contract is single-source; multi-source execution is deferred. |
| Supporting sources | Allow only when they are optional or already supported. | Proceeding while silently ignoring required supporting context would produce misleading answers. |
| Retry | Retry source intake once only for extraction-shaped failures. | This is bounded, cheap, and localized to source intake. |
| Local paths | Trust only explicit local hint fields. | Raw chat text must not grant filesystem access. |
| Provider canonicalization | Use provider grammar, not language-specific stripping. | Percent-encoded and Unicode-safe URLs must remain valid; GitHub repo grammar proves where a repo URL ends. |

## Contracts And Data Shapes

### Internal Source-Intake Proposal

Create this internal module:

```text
src/kazusa_ai_chatbot/coding_agent/code_fetching/source_intake.py
```

The source-intake LLM receives:

```python
{
    "task_text": str,
    "visible_source_spans": list[str],
    "retry_feedback": list[str],
}
```

`visible_source_spans` is a deterministic syntactic aid, not a final source decision. It may include HTTP-like spans, markdown-link URLs, `owner/repo`-like text, package-scheme tokens, and obvious inline-code markers. The LLM assigns semantic roles; deterministic code validates the syntax.

The deterministic span scanner must stay syntax-only. It may expose candidate text to the LLM and to the resolver for anchoring checks, but it must not classify a source as primary, supporting, comparable, optional, required, trusted, or safe.

The LLM returns strict JSON:

```python
{
    "task_source_mode": (
        "single_primary | compare_sources | source_free | unclear"
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
                "archive_url | paste_or_gist | inline_code | attachment | "
                "unknown_url | unknown"
            )
        }
    ]
}
```

Normalizer rules:

- Unknown `task_source_mode`, `role`, or `family_hint` values normalize to `unclear` or `unknown`.
- Non-list `source_mentions` normalizes to an empty list.
- Each `raw_text` is bounded and stripped only for surrounding whitespace.
- Each `raw_text` must match original task text exactly after surrounding-whitespace trimming, or match a deterministic visible span derived from original task text. URL decoding, Unicode normalization, or provider canonicalization must not be used to justify an invented candidate.
- If the LLM omits a visible source span, deterministic code may request one retry with focused feedback. After that retry, omitted spans become `source_intake_failed` or another typed terminal outcome according to the resolver rules.
- The prompt must say: extract only text present in the task, do not invent URLs, preserve percent-encoding, and keep trailing prose out of `raw_text`.

### Internal Resolver Outcome

Create this internal deterministic resolver module:

```text
src/kazusa_ai_chatbot/coding_agent/code_fetching/source_resolver.py
```

The resolver returns an internal outcome with one of these issue codes when it cannot select a source:

```text
no_source_found
source_intake_failed
malformed_source
unsupported_provider
unsupported_source_family
ambiguous_primary_source
unsupported_multi_source
not_accessible
not_found
private_or_auth_required
rate_limited
network_transient
unsafe_source
too_large_or_over_budget
ref_not_found
path_not_found
required_supporting_source_unsupported
extraction_likely_wrong
```

These codes are internal constants used to produce existing public `status`, `message`, `limitations`, and `trace_summary` fields. They do not create a new public result shape.

### Public Mapping And Error Sanitization

Map resolver issue codes into existing public statuses as follows:

| Internal outcome | Public status | Public surface |
|---|---|---|
| `no_source_found`, `ambiguous_primary_source`, `required_supporting_source_unsupported`, `source_intake_failed`, `malformed_source`, `not_accessible`, `not_found`, `private_or_auth_required`, `ref_not_found`, `path_not_found` | `needs_user_input` | Ask for one explicit supported code source, corrected URL, accessible public repository, valid ref, valid path, or clarified required source. |
| `unsupported_provider`, `unsupported_source_family`, `unsafe_source`, `unsupported_multi_source`, `too_large_or_over_budget` | `rejected` | Explain the specific unsupported, unsafe, or over-budget source class in `limitations`. |
| `rate_limited`, `network_transient` | `failed` | Explain that supported-source access failed transiently and can be retried later. |
| `extraction_likely_wrong` | internal retry gate | Retry once, then map the final resolver outcome. |

Clone, raw-download, and access-check failures must be classified from safe exception types or safe known output patterns only. Public `message`, `limitations`, and `trace_summary` must not include raw git stderr, credentials, local absolute cache paths, temporary paths, environment variables, or full command lines.

### Source Family Policy

| Source family | Example | Resolver behavior in this plan |
|---|---|---|
| GitHub repository URL | `https://github.com/a/b` | Supported primary code source. |
| GitHub tree/blob URL | `https://github.com/a/b/tree/main/src` | Supported scoped code source after path validation. |
| Raw GitHub file URL | `https://raw.githubusercontent.com/a/b/main/x.py` | Supported single-file managed download. |
| GitHub `owner/repo` hint | `a/b` | Supported only from explicit `repo_hint` or LLM source mention validated as repository hint. |
| Explicit local root/path field | `local_root_hint`, `local_path_hint` | Supported trusted input. |
| Local path in user text | `C:\repo\file.py` | Unsupported from raw text; ask for explicit trusted source field or repository URL. |
| GitHub issue/PR/discussion URL | `/issues/1`, `/pull/2` | Recognized. Treat as target content by default when the task says "this issue", "this PR", "this discussion", "the linked change", or similar. Derive the repository only when the task explicitly asks for repository/codebase/project analysis and the issue/PR is clearly ancillary. |
| GitHub release/archive URL | `/releases`, `.zip`, `.tar.gz` | Recognized unsupported family. |
| Gist/paste URL | `gist.github.com`, paste sites | Recognized unsupported family. |
| URL shortener or redirector | `t.co`, `bit.ly`, other redirect hosts | Recognized unsupported provider; do not follow redirects in this plan. |
| GitLab/Bitbucket/other repo URL | provider repo URL | Recognized unsupported provider unless a provider fetcher is added in a later plan. |
| Documentation URL | docs page | Supporting source only; required supporting docs cause `required_supporting_source_unsupported`. |
| Package reference | `npm:react`, `pypi:foo`, registry package URL | Recognized unsupported family only when the source form is explicit through a scheme, registry URL, package-manager notation, or wording that asks to analyze a package as the code source. Bare names alone remain ordinary task text. |
| Inline code block | fenced code | Recognized source-free or unsupported for code_fetching; snippet reading is deferred. |
| Attachment | typed adapter attachment metadata | Recognized only if already present in structured request fields; attachment source fetching is deferred. |
| IDN or Unicode host URL | internationalized host or punycode host | Normalize host only for provider recognition; unsupported hosts remain unsupported providers. Preserve path text and percent-encoding. |
| Percent-encoded non-GitHub URL | `https://zh.moegirl.org.cn/%E6%9D%8F...` | Valid URL syntax, unsupported provider for code fetching. |

### Mixed-Source Selection Rules

The resolver applies these rules after normalization and validation:

```text
0 recognized source mentions:
  needs_user_input with no_source_found.

0 valid supported sources + recognized source mentions:
  map the strongest deterministic issue code through the public status
  mapping table.

1 valid primary code source:
  fetch it.

1 valid primary code source + optional reference/supporting sources:
  fetch primary; include unsupported optional sources as limitations, and propagate those limitations through the final coding-agent response.

1 valid primary code source + required unsupported supporting source:
  needs_user_input with required_supporting_source_unsupported.

multiple valid sources from the same repository:
  choose a single scope only when semantic roles make one scope primary
  or when one scoped file/path is clearly nested inside a broader
  supporting repository scope.
  same-priority scopes, different refs, or competing primary scopes need
  user input.

multiple valid repositories + compare_sources mode:
  rejected with unsupported_multi_source.

multiple valid repositories + single_primary or unclear mode:
  needs_user_input with ambiguous_primary_source.

issue/PR URL only:
  derive repo only for explicit repository/codebase/project analysis tasks.
  issue/PR target-content tasks return unsupported_source_family.
  otherwise return unsupported_source_family.

explicit local hint + question URL:
  explicit trusted local hint remains primary only when question sources are
  absent, optional support, or verifiably same-repo context.
  if a question source names a different repository, or if same-repo
  comparison cannot be verified, return needs_user_input.

explicit source_url or repo_url + question URL:
  explicit trusted source field remains primary when question sources are
  absent, optional support, reference-only, or verifiably same-repo context.
  if question text names a different primary or comparison code source,
  return needs_user_input.
  if that explicit field is invalid, unsafe, unsupported, or inaccessible,
  return its typed outcome instead of switching to a question URL.

only inline code or only attachment references:
  return unsupported_source_family or source_free according to operation
  context; do not report generic no_source_found.
```

### Retry Rules

The resolver may ask source intake to retry once for:

```text
extraction_likely_wrong
malformed_source with recoverable original text
visible URL-like span omitted from source_mentions
candidate appears to include trailing natural-language prose
candidate contradicts a stronger explicit source clue in the same task
candidate is unanchored, invented, or only partially copied from task text
```

The resolver must not retry for:

```text
unsupported_provider
unsupported_source_family
private_or_auth_required
not_found
rate_limited
network_transient
unsafe_source
too_large_or_over_budget
unsupported_multi_source
ambiguous_primary_source
explicit trusted source field failed deterministic validation
```

After one retry, deterministic resolver output is final. A visible deterministic span that the LLM still omits after retry must become a typed failure or limitation; resolver code must not promote that span into a source candidate on its own.

## LLM Call And Context Budget

Before:

- Explicit/direct code-fetching source resolution: 0 LLM calls.
- Background coding task before reading/writing: 1 `CODING_AGENT_PM_LLM` call for operation routing.
- Code reading after fetching: existing reading PM/programmer/synthesis calls.

After:

- Explicit `source_url`, `repo_url`, `repo_hint`, `local_root_hint`, or `local_path_hint`: 0 new LLM calls.
- Question-only source mentions: 1 new `CODING_AGENT_PM_LLM` source-intake call inside `code_fetching`.
- Repairable extraction failure: at most 1 additional `CODING_AGENT_PM_LLM` source-intake call.
- Source-free writing tasks: 0 source-intake calls because they do not call `code_fetching`.

Context cap:

- Use the existing `BACKGROUND_WORK_INPUT_CHAR_LIMIT` upstream cap for background task text.
- Bound source-intake `task_text` to the same body size currently accepted by `CodeFetchingRequest.question`.
- Bound `visible_source_spans` to a small list of syntactic candidates, capped by count and per-item length.
- Keep source-intake output below 1200 completion tokens by using a dedicated `LLMCallConfig` with the existing `CODING_AGENT_PM_LLM` route and a local stage-specific completion cap.

Latency impact:

- Adds one PM-route call only for messy question-text source extraction.
- Does not add calls for explicit direct API source fields.
- The background worker path is already asynchronous; this change does not affect live chat response latency.

Verification:

- Add prompt-render and schema-normalization tests.
- Add real LLM source-intake tests for normal, ambiguous, unsupported, and captured-failure cases.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_intake.py`
  - LLM prompt, config, invocation, normalization, and optional retry input for source mention extraction.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_resolver.py`
  - Internal source candidate normalization, candidate anchoring, source family classification, provider-aware canonicalization, unsupported-host no-probe behavior, cardinality policy, retry gating, public status mapping, error sanitization, and mapping into existing GitHub/local source selection.
- `tests/test_coding_agent_source_intake.py`
  - Patched and real LLM tests for the source-intake prompt and normalizer.
- `tests/test_coding_agent_source_resolution.py`
  - Deterministic resolver tests for source families, mixed-source rules, provider-aware URL boundaries, anchoring, unsupported-host no-probe behavior, failure taxonomy, public status mapping, sanitization, and retry gating.
- `tests/fixtures/coding_agent_source_intake_signoff_cases.json`
  - Fixed 20-case final sign-off matrix for source-intake LLM plus deterministic resolver integration.
- `tests/test_coding_agent_source_intake_signoff.py`
  - Fixture integrity test and live one-case-at-a-time sign-off harness using the fixture.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py`
  - Replace question-text URL extraction in `_select_github_source()` with the source-intake/resolver path.
  - Preserve explicit local hint handling before remote source resolution.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/github.py`
  - Add provider-aware canonicalization helpers where GitHub grammar proves a trailing token is not part of the repo/source.
  - Preserve percent-encoded URL paths and avoid language-specific stripping.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/source_scope.py`
  - Keep single-source selection semantics; update only if resolver needs a clearer public message for conflicting same-repo scopes.
- `src/kazusa_ai_chatbot/coding_agent/code_fetching/README.md`
  - Document the source-intake stage, recognized source families, typed outcomes, and single-source execution policy.
- `src/kazusa_ai_chatbot/coding_agent/README.md`
  - Update the architecture diagram and text so `code_fetching` shows source intake before deterministic source resolution.
- `tests/test_coding_agent_fetching.py`
  - Add regression coverage for captured production-style source text, percent-encoded unsupported URLs, same-repo scope specificity, and unsupported mixed-source inputs.
- `tests/test_coding_agent_interface.py`
  - Add patched handoff coverage proving background code-reading requests reach `code_fetching` and that source-intake success/failure maps to the current response shape.
- `tests/test_background_work_coding_agent.py`
  - Update or add worker mapping assertions only if new limitations/trace labels change worker metadata expectations.

### Keep

- `src/kazusa_ai_chatbot/background_work/router.py`
  - Remains route-only.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`
  - Keeps the same public call into `handle_background_coding_task()` unless tests reveal a necessary metadata mapping update.
- `src/kazusa_ai_chatbot/coding_agent/models.py`
  - Public request and response contracts remain unchanged.
- `src/kazusa_ai_chatbot/coding_agent/code_reading/*`
  - Reading PM and programmer behavior remain unchanged.
- `src/kazusa_ai_chatbot/coding_agent/code_writing/*`
  - Source-free writing behavior remains unchanged.

## Overdesign Guardrail

- Actual problem: raw user task text can contain messy, mixed, or unsupported source mentions, and the current deterministic question URL extraction produces generic or wrong failures.
- Minimal change: add one source-intake specialist and one deterministic resolver inside `code_fetching`, while keeping public request/response shapes and existing fetchers.
- Ownership boundaries: LLM extracts semantic source mentions and roles; deterministic code validates source truth, safety, access, cardinality, retry eligibility, and final selection.
- Rejected complexity: durable source schema, background-work payload changes, new model route, generic web browsing, package lookup, PR/issue fetching, attachment source fetching, multi-source reading, broad retry loops, compatibility shims, and provider plugin architecture.
- Evidence threshold: add a new source fetcher or multi-source execution only after a user-approved plan names the provider, fetch contract, safety policy, tests, and user-visible behavior.

## Plan Review Findings Addressed

| Finding | Resolution in this plan |
|---|---|
| Bare package names could turn ordinary task text into unsupported-source failures. | Package references require explicit package syntax, registry URLs, package-manager notation, or task wording that makes the package the source. |
| Deterministic visible spans could become semantic pre-processing. | Visible spans are syntax-only observation aids and anchoring inputs; they cannot assign roles or promote candidates. |
| LLM source extraction could invent or rewrite URLs. | Every `raw_text` candidate must be anchored to original task text or a visible span, with no URL-decoding-based justification. |
| Issue and PR URLs could be misread as repositories. | Issue/PR links are target content by default; repository derivation requires explicit repository/codebase/project wording and ancillary issue/PR role. |
| Unsupported hosts and redirectors could create unsafe probing behavior. | Resolver access checks run only for supported providers and trusted local fields; redirectors and unsupported hosts are not fetched. |
| Explicit source fields could be masked by a different source in question text. | Explicit fields are authoritative; their deterministic failures become typed outcomes instead of silent substitution. |
| Local hint plus remote text could select the wrong repository. | Local hints remain primary only for absent, optional, or verifiably same-repo question sources; conflicts require user input. |
| Same-repository scopes lacked a conflict definition. | Selection now requires one primary semantic scope or a nested broader support scope; competing scopes, same-priority scopes, and different refs require user input. |
| Access failures could leak raw command output or local paths. | Public error mapping now requires safe classification and forbids raw stderr, credentials, absolute cache paths, temp paths, env vars, and command lines. |
| Optional unsupported context could disappear from the final answer. | Optional unsupported supporting sources must propagate as limitations through the final coding-agent response. |

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve this plan's contracts.
- The responsible agent must not introduce new architecture, alternate cutover strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside `code_fetching`, coding-agent README documentation, and focused tests as high-scrutiny changes.
- Updating an existing module outside the target module or introducing a new prompt, variable, or code path requires strong justification in this plan before implementation.
- If equivalent validation or parser behavior already exists, the responsible agent must reuse, move, or locally adapt it instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If this plan and code disagree, the responsible agent must preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Add deterministic resolver contract tests.
   - File: `tests/test_coding_agent_source_resolution.py`.
   - Cover source family classification, GitHub canonicalization, mixed-source cardinality, unsupported provider/family outcomes, and retry eligibility.
   - Run before implementation and record the expected missing-module failure.
2. Add source-intake prompt/normalizer tests.
   - File: `tests/test_coding_agent_source_intake.py`.
   - Use patched LLM output for normalizer and retry feedback.
   - Add real LLM cases marked `live_llm` for later one-at-a-time execution.
3. Implement `source_intake.py`.
   - Add prompt, config using `CODING_AGENT_PM_LLM`, invocation, parsing, normalization, and bounded retry input.
   - Keep prompt static except current-run human payload.
4. Implement `source_resolver.py`.
   - Add internal candidate normalization, candidate anchoring, issue codes, family classification, GitHub canonicalization, mixed-source selection, unsupported-host no-probe behavior, public status mapping, error sanitization, and retry gating.
5. Wire `agent.py`.
   - Preserve local hint precedence.
   - Preserve explicit field behavior.
   - Replace raw question URL extraction with source-intake plus resolver for question text.
   - Map resolver terminal outcomes into existing `CodeFetchingResult` fields.
6. Update GitHub helper behavior.
   - Add grammar-aware canonicalization for GitHub repo/tree/blob/raw candidates.
   - Preserve percent-encoding and avoid language-specific stripping.
7. Add integration tests.
   - Extend `tests/test_coding_agent_fetching.py` for captured production-style text and mixed families.
   - Extend `tests/test_coding_agent_interface.py` for background code-reading handoff through fetching and limitation propagation.
   - Update `tests/test_background_work_coding_agent.py` only if worker metadata expectations change.
8. Add final sign-off matrix harness.
   - File: `tests/fixtures/coding_agent_source_intake_signoff_cases.json`.
   - File: `tests/test_coding_agent_source_intake_signoff.py`.
   - Validate fixture count, required metadata, anti-cheat contract, one-case selection, trace artifact creation, and hard gates for expected status, issue code, source literal preservation, selected primary source, unsupported-host no-probe behavior, limitations, and forbidden failure modes.
9. Update ICD docs.
   - Update `code_fetching/README.md`.
   - Update `coding_agent/README.md` architecture text/diagram.
10. Run deterministic verification.
11. Run real LLM source-intake and final sign-off matrix tests one at a time and inspect logs.
12. Run independent code review and remediate findings inside this change surface.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused resolver and intake test contracts established.
  - Covers: implementation steps 1-2.
  - Verify: run focused tests and record expected missing-module or failing-contract output.
  - Evidence: record test command outputs in `Execution Evidence`.
  - Handoff: next agent starts Stage 2.
  - Sign-off: pending.
- [ ] Stage 2 - source-intake and deterministic resolver implemented.
  - Covers: implementation steps 3-6.
  - Verify: focused resolver and intake patched tests pass.
  - Evidence: record changed production files and focused test output.
  - Handoff: next agent starts Stage 3.
  - Sign-off: pending.
- [ ] Stage 3 - code-fetching and background integration tests complete.
  - Covers: implementation step 7.
  - Verify: focused fetching/interface/background tests pass.
  - Evidence: record test outputs and any baseline changes.
  - Handoff: next agent starts Stage 4.
  - Sign-off: pending.
- [ ] Stage 4 - ICD documentation updated.
  - Covers: implementation step 9.
  - Verify: README greps and documentation consistency checks pass.
  - Evidence: record changed docs and grep outputs.
  - Handoff: next agent starts Stage 5.
  - Sign-off: pending.
- [ ] Stage 5 - deterministic and live LLM verification complete.
  - Covers: implementation steps 8, 10, and 11.
  - Verify: deterministic tests pass; each real LLM case and each final sign-off matrix case is run one at a time and inspected.
  - Evidence: record commands, trace artifact paths, and human judgment for each live LLM case.
  - Handoff: next agent starts Stage 6.
  - Sign-off: pending.
- [ ] Stage 6 - independent code review complete.
  - Covers: implementation step 12.
  - Verify: review findings are recorded, fixes are applied inside scope, and affected tests are rerun.
  - Evidence: record review outcome, remediation, rerun commands, and residual risks.
  - Handoff: plan can move to completion only after this stage passes.
  - Sign-off: pending.

## Verification

### Static Checks

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\coding_agent\code_fetching\source_intake.py src\kazusa_ai_chatbot\coding_agent\code_fetching\source_resolver.py`
  - Expected: exit code 0.
- `rg "extract_http_urls\\(question\\)" src\kazusa_ai_chatbot\coding_agent`
  - Expected: no matches after cutover. A nonzero `rg` exit code is acceptable for no matches.
- `rg "BACKGROUND_WORK_LLM|CODING_AGENT_PROGRAMMER_LLM" src\kazusa_ai_chatbot\coding_agent\code_fetching`
  - Expected: no matches except documentation or comments explicitly stating those routes are not used.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_source_resolution.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_fixture_contract -q`
- `venv\Scripts\python -m pytest tests\test_coding_agent_interface.py::test_handle_background_coding_task_routes_to_reading -q`
- `venv\Scripts\python -m pytest tests\test_background_work_coding_agent.py -q`

Required resolver and integration cases:

- GitHub repository URL followed by CJK punctuation and prose resolves to the canonical repository.
- Percent-encoded and Unicode URL text is preserved and classified by provider support, not by language-specific stripping.
- Bare package-like words inside normal task text do not become source candidates.
- Explicit package-manager notation and registry URLs become unsupported package references.
- Issue/PR target-content tasks return unsupported source-family outcomes.
- Explicit repository/codebase/project tasks with ancillary issue/PR links derive the repository.
- URL shorteners, redirectors, unsupported hosts, and generic document URLs are classified without network probing.
- Malformed, misspelled, not-found, private, invalid-ref, and invalid-path supported-provider sources return specific `needs_user_input` outcomes.
- LLM candidates that are invented, rewritten, or unanchored trigger the single retry only when retry-eligible; otherwise they become typed failures.
- Invalid explicit `source_url` or `repo_url` failures are not replaced by question-text sources.
- Explicit local hints with conflicting remote repository text return `needs_user_input`.
- Same-repository scopes select only a semantically primary or nested scope; same-priority scopes and different refs require user input.
- Optional unsupported supporting sources propagate as final response limitations.
- Required unsupported supporting sources block execution.
- Inline-code-only and attachment-only inputs produce source-free or unsupported-family outcomes, not generic no-source failures.
- Multiple valid repositories and mixed source families follow the single-source, compare, or ambiguity policy.

### Patched LLM Tests

- `venv\Scripts\python -m pytest tests\test_coding_agent_source_intake.py -q -m "not live_llm"`
  - Expected: patched LLM source-intake normalizer and retry-gating tests pass.

### Real LLM Tests

Run one case at a time with output inspected:

- `venv\Scripts\python -m pytest tests\test_coding_agent_source_intake.py::test_live_source_intake_extracts_captured_github_task -q -s -m live_llm`
- `venv\Scripts\python -m pytest tests\test_coding_agent_source_intake.py::test_live_source_intake_marks_unsupported_web_url -q -s -m live_llm`
- `venv\Scripts\python -m pytest tests\test_coding_agent_source_intake.py::test_live_source_intake_marks_multiple_repos_mode -q -s -m live_llm`

Expected: each case produces parseable JSON, preserves source literals supplied by the input, assigns a plausible role/family, and avoids invented URLs. The agent must inspect logs before counting the case as passed.

### E2E Regression

- Add or update one deterministic background coding regression using the captured production-style task:
  - task text contains `https://github.com/sdyzjx/open-yachiyo` followed by natural-language punctuation/prose.
  - managed checkout is patched to avoid network.
  - expected: background coding path reaches a successful resolved repository/source scope or the direct code-fetching result succeeds.

### Final Sign-Off Intake Resolver Matrix

Fixture:

- `tests/fixtures/coding_agent_source_intake_signoff_cases.json`

Harness:

- `tests/test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_case_live_llm`

Run one case at a time with output inspected. Do not batch the 20 live cases.

Command template:

```powershell
$env:CODING_AGENT_SOURCE_INTAKE_SIGNOFF_CASE_ID='<case_id>'
venv\Scripts\python -m pytest tests\test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_case_live_llm -q -s -m live_llm
```

Required sign-off cases:

| Case | Use case | Expected terminal outcome |
|---|---|---|
| `csi_001_github_trailing_cjk_prose` | Captured GitHub URL with CJK punctuation and trailing prose | `succeeded` repository |
| `csi_002_markdown_repo_link` | Markdown GitHub repository link | `succeeded` repository |
| `csi_003_github_tree_directory_scope` | GitHub tree URL scoped to a directory | `succeeded` directory |
| `csi_004_github_blob_file_anchor` | GitHub blob URL with line fragment | `succeeded` file |
| `csi_005_raw_github_file` | Raw GitHub file URL | `succeeded` file |
| `csi_006_owner_repo_hint` | `owner/repo` source hint in prose | `succeeded` repository |
| `csi_007_percent_encoded_non_github_url` | Percent-encoded non-GitHub URL | `rejected` unsupported provider |
| `csi_008_unsupported_hosts_no_probe` | GitLab and URL shortener source mentions | `rejected` unsupported provider |
| `csi_009_explicit_package_reference` | Explicit package-manager source reference | `rejected` unsupported source family |
| `csi_010_bare_package_word_not_source` | Bare package word without source syntax | `needs_user_input` no source |
| `csi_011_github_issue_target_content` | GitHub issue as target content | `rejected` unsupported source family |
| `csi_012_github_issue_ancillary_repo_analysis` | GitHub issue used only to identify repo | `succeeded` repository |
| `csi_013_multi_repo_compare_unsupported` | Explicit comparison between two repositories | `rejected` unsupported multi-source |
| `csi_014_multi_repo_ambiguous_primary` | Two repositories without a primary | `needs_user_input` ambiguous primary |
| `csi_015_same_repo_nested_scope` | Same repo plus nested directory focus | `succeeded` directory |
| `csi_016_same_repo_conflicting_files` | Same repo with competing file scopes | `needs_user_input` ambiguous primary |
| `csi_017_required_supporting_docs_unsupported` | Required docs plus supported repo | `needs_user_input` required support unsupported |
| `csi_018_optional_supporting_docs_limitation` | Optional docs plus supported repo | `succeeded` with limitation |
| `csi_019_raw_local_path_in_chat` | Raw local filesystem path in chat text | `rejected` unsupported source family |
| `csi_020_explicit_invalid_source_authoritative` | Invalid explicit source plus valid question URL | `needs_user_input` malformed explicit source |

For every case, the trace artifact must record case id, task text, request fields, visible spans, raw source-intake output, normalized source mentions, retry feedback if used, resolver outcome, final public result, no-probe host evidence, and human inspection notes. Final plan sign-off is blocked until all 20 cases pass their hard gates and the recorded inspection says the behavior satisfies the case contract.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, prompt payload leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused and regression tests, real LLM trace artifacts, execution evidence, and path-safe Windows commands.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture/documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `code_fetching` contains a source-intake LLM specialist and deterministic resolver inside its own package.
- Direct explicit source requests continue to resolve without an added source-intake LLM call.
- Question-only coding tasks with a clean GitHub repo URL resolve.
- Question-only coding tasks with a GitHub repo URL followed by natural-language punctuation/prose resolve to the canonical GitHub repo.
- Percent-encoded non-GitHub URLs remain syntactically intact and are rejected as unsupported providers, not malformed Chinese or Unicode URLs.
- LLM-proposed source candidates are anchored to original task text or deterministic visible spans.
- Deterministic visible spans never select source roles without source-intake semantics.
- Unsupported hosts, URL shorteners, generic document URLs, paste sites, and package registries are not probed by network access checks.
- Multiple code sources return `needs_user_input` for ambiguity or `rejected` with `unsupported_multi_source` for explicit compare/multi-source requests.
- Same-repository scope conflicts require clarification unless one source is semantically primary or nested under a broader support scope.
- Issue/PR target-content tasks are not silently converted into default-branch repository analysis.
- Invalid explicit source fields return their own typed outcomes instead of switching to a question-text source.
- Unsupported providers and unsupported source families return specific limitations instead of generic no-source failure.
- Malformed, wrong, inaccessible, private, invalid-ref, and invalid-path supported-provider sources return specific corrective `needs_user_input` outcomes.
- Required unsupported supporting context blocks the task with a specific limitation instead of silently proceeding.
- Optional unsupported supporting context appears in final response limitations.
- Local filesystem paths from raw user text are not treated as trusted local hints.
- Raw clone/download/access-check errors are sanitized before entering public messages, limitations, or traces.
- Existing source-fetching, background-worker, and coding-agent interface tests pass.
- Real LLM source-intake cases are run one at a time and inspected.
- All 20 final sign-off intake resolver matrix cases in `tests/fixtures/coding_agent_source_intake_signoff_cases.json` pass one at a time with trace artifacts and human inspection notes.
- The coding-agent and code-fetching ICDs document the implemented architecture.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Source-intake LLM invents URLs | Prompt forbids invention; deterministic resolver validates raw text against original task spans. | Patched tests and live LLM literal-preservation cases. |
| Deterministic code overrides semantic intent | Resolver validates and selects only after LLM role extraction; ambiguous role cases ask for clarification. | Mixed-source and no-prepost review tests. |
| Supporting docs are ignored | Required unsupported supporting sources block the task. Optional references become limitations. | Required-supporting-source tests. |
| GitHub canonicalization strips valid URL content | Provider grammar trims only invalid GitHub owner/repo/scope tails; percent-encoded URLs are preserved. | Percent-encoded URL and CJK punctuation regression tests. |
| Added LLM call hurts background latency | Source-intake runs only for question-text source mentions and at most retries once. | LLM budget review and live test timing notes. |
| Plan expands into generic web/source ingestion | Deferred section forbids new provider fetchers, web browsing, packages, archives, attachments, and multi-source reading. | Independent code review and static greps. |

## Execution Evidence

- 2026-07-06: User explicitly requested execution without subagent. Running
  under single-agent fallback execution for this plan.
- 2026-07-06: Added source-intake, deterministic resolver, fixture contract,
  20-case signoff fixture, and direct code-fetching captured-task regression.
- 2026-07-06: Initial red test run failed at missing `source_intake` module,
  establishing the planned test boundary before production implementation.
- 2026-07-06: Final static checks passed:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\coding_agent\code_fetching\source_intake.py src\kazusa_ai_chatbot\coding_agent\code_fetching\source_resolver.py tests\test_coding_agent_source_intake.py tests\test_coding_agent_source_resolution.py tests\test_coding_agent_source_intake_signoff.py`
  - `rg "extract_http_urls\\(question\\)" src\kazusa_ai_chatbot\coding_agent` returned no matches.
  - `rg "BACKGROUND_WORK_LLM|CODING_AGENT_PROGRAMMER_LLM" src\kazusa_ai_chatbot\coding_agent\code_fetching` returned no matches.
- 2026-07-06: Final deterministic tests passed:
  - `venv\Scripts\python -m pytest tests\test_coding_agent_source_resolution.py tests\test_coding_agent_source_intake.py -q -m "not live_llm"`: 17 passed, 3 deselected.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_fixture_contract -q`: 1 passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_fetching.py -q`: 30 passed.
  - `venv\Scripts\python -m pytest tests\test_coding_agent_interface.py::test_handle_background_coding_task_routes_to_reading -q`: 1 passed.
  - `venv\Scripts\python -m pytest tests\test_background_work_coding_agent.py -q`: 3 passed.
- 2026-07-06: Final live LLM spot checks passed one at a time:
  - `test_live_source_intake_extracts_captured_github_task`
  - `test_live_source_intake_marks_unsupported_web_url`
  - `test_live_source_intake_marks_multiple_repos_mode`
- 2026-07-06: Full 20-case final signoff matrix passed one case at a time
  from the final prompt state using
  `tests/test_coding_agent_source_intake_signoff.py::test_source_intake_signoff_case_live_llm`.
  Trace artifacts were written under `test_artifacts/llm_traces`.
- 2026-07-06: `git diff --check` passed with line-ending warnings only.
- 2026-07-06: Final line-length scan over changed Python files passed with no
  lines over 88 characters after wrapping the touched CJK test string.
- 2026-07-06: Self-review follow-up added source-access failure classification
  for managed GitHub clone/raw-download failures, changed missing resolved
  scopes to `needs_user_input`, and reran the affected fetching suite.
- 2026-07-06: Added and ran a patched handoff test for the source-intake
  feedback loop:
  `tests/test_coding_agent_source_resolution.py::test_select_source_retries_source_intake_with_resolver_feedback`.
  The test verifies the deterministic resolver rejects an unanchored first
  intake result, passes retry feedback into the second intake call, accepts the
  corrected anchored source, and records `source_intake:retried_once`.
- 2026-07-06: Independent subagent review was not run because the user
  explicitly required execution without subagents. Performed single-agent
  self-review against ownership boundaries, prompt/contract behavior, explicit
  source precedence, unsupported-provider no-probe behavior, and verification
  evidence.
