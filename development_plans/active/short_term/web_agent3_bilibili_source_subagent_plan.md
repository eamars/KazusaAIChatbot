# web_agent3 bilibili source subagent plan

## Summary

- Goal: Add Bilibili as a real read-only `web_agent3` source subagent backed
  by `bilibili-api-python`.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`
- Overall cutover strategy: bigbang for the Bilibili source. Add one final
  `bilibili` source module with no legacy placeholder fallback.
- Highest-risk areas: optional dependency discovery, import-time failure when
  the dependency is absent, prompt roster drift, disabled-source fallback to
  generic web search, Bilibili API response shape drift, and over-broad
  metadata leakage.
- Acceptance criteria: Bilibili registers only when `bilibili_api` is
  importable, supports only `read` and `search`, reads Bilibili video URLs or
  ids through source-local parsing, searches Bilibili videos with bounded
  result compaction, and preserves the existing `WebAgent3().run(...)` public
  contract.

## Context

`web_agent3` now discovers source modules from `subagent/*.py`. Available
sources expose `SOURCE`, `DESCRIPTION`, `SUPPORTED_ACTIONS`, and
`execute(decision)`. Optional sources expose `is_enabled()` and stay out of
the router prompt when unavailable.

The user wants Bilibili support for read-only evidence requests such as:

- `https://www.bilibili.com/video/BV1CqV266EJY/ 讲了什么内容`
- `帮我在bilibili上搜索关于vibe coding相关视频并且推荐给我一个最热门的视频`

Research findings:

- `bilibili-api-python` is published on PyPI as version `17.4.2`, with
  project import package `bilibili_api`.
- `bilibili_api.video.Video` accepts `bvid` or `aid` and exposes
  `get_info()`, `get_pages()`, `get_related()`, and `get_subtitle(cid)`.
- `bilibili_api.search.search_by_type(...)` supports video search with
  `SearchObjectType.VIDEO` and video ordering through `OrderVideo`, including
  `OrderVideo.CLICK` for most-clicked results.
- The current project virtual environment does not have `bilibili_api`
  installed, so Bilibili must be dependency-gated and unavailable by default
  unless the optional dependency is installed.

This plan follows the Hermes-style pattern discussed with the user:
provider-specific API code lives in the source module, provider-specific
dependencies are optional project extras, and missing optional dependencies
disable the provider instead of breaking process import.

## Mandatory Skills

- `development-plan`: load before editing, executing, reviewing, or signing
  off this plan.
- `local-llm-architecture`: load before changing source routing, prompt source
  descriptions, graph behavior, or LLM-facing Bilibili descriptions.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files containing CJK prompt or
  description strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, the parent or active execution agent must
  reread this entire plan before continuing implementation, verification,
  handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use parent-led native subagent execution. If native subagent capability is
  unavailable, stop before production implementation unless the user explicitly
  approves fallback execution.
- Use `venv\Scripts\python.exe` for Python commands.
- Do not read the real `.env` file during implementation or verification.
- Keep `WebAgent3().run(task, context, max_attempts)` unchanged.
- Keep router output shape exactly `action`, `source`, and `query`.
- Keep Bilibili source execution, id parsing, API parameter construction,
  request execution, limits, error observations, and result compaction inside
  `subagent/bilibili.py`.
- Keep Bilibili source read-only. Do not add login, cookie, credential,
  favorite, like, comment, upload, playback, download, or account actions.
- Keep `bilibili_api` imports lazy inside execution or availability helpers so
  source discovery works when the optional dependency is absent.
- Declare `bilibili-api-python` as a project optional dependency. Do not add it
  to core dependencies.
- Do not add runtime lazy installation in this plan.
- Do not add `.env` configuration for Bilibili in this plan.
- Do not use `web_search` or `web_read` as a hidden Bilibili fallback path.
- Do not add live external-service tests to the regular deterministic suite.

## Must Do

- Add a `bilibili` optional dependency extra to `pyproject.toml`.
- Add `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py`.
- Register Bilibili only when `bilibili_api` is importable.
- Expose `SOURCE = "bilibili"` and `SUPPORTED_ACTIONS = ("read", "search")`.
- Add a Bilibili `DESCRIPTION` that gives source-local query generation rules
  for video URL/id reads and Bilibili video searches.
- Parse Bilibili video URLs, BV ids, and AV ids inside `bilibili.py`.
- Implement `read` using `bilibili_api.video.Video`.
- Implement `search` using `bilibili_api.search.search_by_type` with
  `SearchObjectType.VIDEO`.
- Use `OrderVideo.CLICK` for search requests that ask for the hottest or most
  popular video.
- Return bounded prompt-safe observations with compact metadata and source
  evidence.
- Update router normalization so an explicit disabled `bilibili` source
  normalizes to `stop` instead of falling back to `web_search`.
- Update tests, ICD, HOWTO, and plan registry.

## Deferred

- Do not add Bilibili login, cookies, credential storage, or account-scoped
  API access.
- Do not fetch or expose video binary downloads, image binaries, danmaku raw
  streams, comments, likes, favorites, history, or account state.
- Do not add browser automation, JavaScript execution, CAPTCHA handling, or
  anti-bot bypass.
- Do not add runtime dependency lazy-installation.
- Do not add Bilibili-specific `.env` settings.
- Do not change L2d, RAG dispatcher, cognition, dialog, adapters,
  persistence, consolidation, scheduler, or database code.
- Do not add Reddit, YouTube, or other site-specific sources.
- Do not reintroduce the previous placeholder Bilibili fallback module.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Bilibili source module | bigbang | Add the final `subagent/bilibili.py` implementation. |
| Optional dependency | bigbang | Declare `bilibili-api-python` under `[project.optional-dependencies]` only. |
| Source discovery | bigbang | Register `bilibili` only when `bilibili_api` is importable. |
| Router fallback | bigbang | Treat explicit disabled `bilibili` as `stop`; do not fall back to `web_search`. |
| Runtime API scope | bigbang | Support read-only metadata/subtitle/search evidence only. |
| Tests/docs | bigbang | Update expectations to include config-free dependency-gated Bilibili availability. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- For bigbang areas, implement the final contract directly and avoid
  compatibility shims, aliases, and hidden fallback paths.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

Final source roster by environment:

```text
always:
  web_read

when SEARXNG_URL is configured:
  web_search

when NHENTAI_TOKEN is configured:
  nhentai

when bilibili_api is importable:
  bilibili
```

`bilibili` supports:

- `read`: video URL, BV id, or AV id.
- `search`: Bilibili video keyword search.

`bilibili` observations are compact evidence packets. They may include title,
video URL, `bvid`, `aid`, uploader display name, description, duration,
publish time, view-like statistics, page metadata, tags when available, and
subtitle-derived text when available. They must include a limitation note when
full spoken-content evidence is unavailable because subtitles are absent.

`bilibili` observations must not include cookies, credentials, account data,
raw headers, raw API blobs, binary media, images, comments, favorites, history,
or download URLs.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Source name | Use `bilibili` | The source name is direct and model-readable. |
| Dependency | Add `bilibili-api-python==17.4.2` as optional extra | The dependency is source-specific and should not affect default installs. |
| Availability | Gate on `importlib.util.find_spec("bilibili_api")` | Missing optional dependency disables the source without importing the SDK. |
| Runtime install | Exclude lazy installation | The project does not currently have a lazy dependency installer pattern. |
| Config | Add no Bilibili `.env` config | Public read/search use cases require no user-specific secret. |
| Read operation | Use `Video(...).get_info()` and `get_pages()` | These are the library's source-specific video metadata APIs. |
| Subtitle evidence | Use `get_subtitle(cid)` when a `cid` is available | Subtitle text is stronger evidence for "讲了什么内容" than metadata alone. |
| Search operation | Use `search_by_type(..., SearchObjectType.VIDEO, OrderVideo.CLICK)` for popularity requests | `OrderVideo.CLICK` is the library's most-clicked video ordering. |
| Disabled explicit source | Normalize explicit disabled `bilibili` to `stop` | A Bilibili-specific request should not silently become generic web search. |
| Final answer | Keep existing finalizer | Generic evidence synthesis can answer from compact source observations. |

## Contracts And Data Shapes

Source module interface:

```python
SOURCE = "bilibili"
SUPPORTED_ACTIONS = ("read", "search")
DESCRIPTION = '''...'''

def is_enabled() -> bool: ...
async def execute(decision: _RouterDecision) -> Any: ...
```

Execution behavior:

```text
read query:
  - Bilibili video URL
  - BV id
  - AV id

search query:
  - Bilibili video search keywords
  - Optional user language such as "最热门" is handled by source-local search
    order selection.
```

Read observation shape:

```python
{
    "status": "success" | "partial" | "error",
    "source": "bilibili",
    "action": "read",
    "query": str,
    "target": {
        "bvid": str | None,
        "aid": int | None,
        "url": str | None,
    },
    "video": {
        "title": str,
        "uploader": str,
        "description": str,
        "duration": str,
        "published_at": str,
        "stats": dict[str, int | str],
        "pages": list[dict[str, str | int]],
        "subtitle_excerpt": str,
        "evidence_basis": "subtitle_and_metadata" | "metadata_only",
    },
    "message": str,
}
```

Search observation shape:

```python
{
    "status": "success" | "partial" | "error",
    "source": "bilibili",
    "action": "search",
    "query": str,
    "order": "click" | "totalrank",
    "results": [
        {
            "rank": int,
            "title": str,
            "bvid": str | None,
            "url": str | None,
            "uploader": str,
            "duration": str,
            "published_at": str,
            "view_count": int | str | None,
            "description": str,
            "recommendation_basis": str,
        }
    ],
    "message": str,
}
```

Failure conditions:

- Missing optional dependency: source is not registered.
- Unsupported action: return bounded `error` observation.
- Missing video id for `read`: return bounded `error` observation.
- API exception or response mismatch: return bounded `error` observation.
- Missing subtitle: return `partial` or `success` with
  `evidence_basis="metadata_only"` and a limitation message.
- Empty search results: return `status="partial"` with empty `results` and a
  message that no candidates were found.

## LLM Call And Context Budget

The number of response-path LLM calls is unchanged.

Affected response-path call:

| Stage | Route | Before | After | Budget |
|---|---|---|---|---|
| `_tool_call_generator` | `WEB_SEARCH_LLM` | Source roster can include `web_read`, configured `web_search`, and configured `nhentai`. | Source roster can additionally include dependency-enabled `bilibili`. | At most `MAX_WEB_SEARCH_AGENT_RETRY` generator calls per `WebAgent3.run`; unchanged. |

Unchanged calls:

| Stage | Route | Change |
|---|---|---|
| `_tool_call_evaluator` | `WEB_SEARCH_LLM` | No prompt, route, retry, or context-budget change. |
| `_tool_call_finalizer` | `WEB_SEARCH_LLM` | No prompt, route, retry, or context-budget change. |

Context inputs:

- System prompt changes only in the generated source roster under source
  descriptions.
- Human payload remains unchanged.
- Bilibili source description must stay compact and process-stable for the
  Python session.
- Conservative static prompt growth is under 2,000 characters, below the
  default 50k-token planning cap.
- Bilibili observations must cap search results and subtitle excerpts so the
  evaluator/finalizer context remains bounded.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py`
  - Real Bilibili source subagent.
- `tests/test_web_agent3_bilibili.py`
  - Deterministic tests for Bilibili parsing, availability, read/search
    execution, and prompt-safe observation compaction.

### Modify

- `pyproject.toml`
  - Add optional dependency extra for Bilibili.
- `development_plans/README.md`
  - Add this plan row.
- `src/kazusa_ai_chatbot/rag/web_agent3/contracts.py`
  - Stop explicit disabled `bilibili` router decisions instead of falling back
    to `web_search`.
- `src/kazusa_ai_chatbot/rag/web_agent3/README.md`
  - Document the Bilibili source and optional dependency behavior.
- `docs/HOWTO.md`
  - Document installation command for Bilibili optional source.
- `tests/test_web_agent3.py`
  - Update source discovery and prompt roster tests to account for optional
    dependency-gated `bilibili`.
- `tests/test_web_agent3_routing.py`
  - Add source/action normalization coverage for enabled and disabled
    `bilibili`.

### Keep

- `WebAgent3().run(...)` public contract.
- Existing `web_read`, `web_search`, and `nhentai` source modules.
- Existing source discovery interface.
- Existing evaluator and finalizer prompts.
- Existing `.env` configuration shape.

## Overdesign Guardrail

- Actual problem: `web_agent3` cannot answer Bilibili-specific video read and
  video search requests with source-specific API evidence.
- Minimal change: add one dependency-gated `bilibili` source module, one
  optional dependency extra, focused deterministic tests, and docs.
- Ownership boundaries: the router selects `bilibili`; `bilibili.py` owns
  Bilibili parsing/API/compaction; deterministic tests own API-shape fixtures;
  the existing finalizer owns visible answer synthesis.
- Rejected complexity: lazy dependency installation, credentials, cookies,
  account APIs, browser automation, downloader APIs, comment/danmaku history,
  new LLM calls, new config variables, source compatibility aliases, and
  generic web fallback for disabled Bilibili.
- Evidence threshold: add credentials, browser fallback, live smoke tests, or
  lazy installation only after a separate approved plan names a concrete
  failing Bilibili use case that public metadata/search APIs cannot satisfy.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when they preserve
  the contracts in this plan.
- The responsible agent must search existing web_agent3 source modules before
  creating helpers and follow the local source-subagent pattern.
- The responsible agent must not add compatibility layers, alternate source
  names, fallback mappers, browser paths, new LLM stages, or extra Bilibili
  capabilities.
- The responsible agent must keep changes outside the listed change surface
  out of scope.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If upstream `bilibili-api-python` response fields differ from the expected
  fixture shape, the agent must compact only fields that are present and record
  the mismatch in `Execution Evidence`.
- If the plan and code disagree, preserve this plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent adds focused deterministic Bilibili tests.
   - Add `tests/test_web_agent3_bilibili.py`.
   - Add tests:
     - `test_bilibili_is_disabled_without_optional_dependency`
     - `test_bilibili_is_enabled_when_optional_dependency_is_importable`
     - `test_bilibili_extracts_bv_and_av_targets`
     - `test_bilibili_read_returns_compact_metadata_and_subtitle_basis`
     - `test_bilibili_read_reports_metadata_only_when_subtitle_missing`
     - `test_bilibili_search_uses_video_click_order_for_hot_request`
     - `test_bilibili_search_returns_bounded_prompt_safe_candidates`
     - `test_bilibili_api_errors_are_bounded`
   - Update `tests/test_web_agent3.py` source discovery subprocess helper to
     support a fake `bilibili_api` package on `PYTHONPATH`.
   - Update `tests/test_web_agent3_routing.py` with enabled and disabled
     `bilibili` normalization cases.
   - Expected before implementation: tests fail because `bilibili.py` and the
     optional dependency extra do not exist.

2. Parent starts one production-code subagent.
   - Scope: `pyproject.toml`,
     `src/kazusa_ai_chatbot/rag/web_agent3/contracts.py`, and
     `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py`.
   - The production-code subagent edits production code only and reports
     changed files, commands run, blockers, and residual risks.

3. Production-code subagent implements packaging and Bilibili module.
   - Add `[project.optional-dependencies].bilibili`.
   - Create `subagent/bilibili.py`.
   - Keep SDK imports lazy.
   - Implement availability, target parsing, read, search, bounded errors, and
     prompt-safe compaction.
   - Update router normalization for explicit disabled `bilibili`.

4. Parent updates docs and integration tests.
   - Update web_agent3 ICD and HOWTO.
   - Update discovery and router prompt tests for dependency-gated Bilibili.
   - Run focused Bilibili and source discovery tests.

5. Parent runs full verification gates.
   - Run static compile, focused tests, web_agent3 regression tests, and static
     greps.
   - Record command summaries in `Execution Evidence`.

6. Parent starts one independent code-review subagent.
   - Review the implementation against this plan, source ownership, dependency
     gating, prompt safety, test determinism, and docs.
   - Parent fixes approved-scope findings and reruns affected checks.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  lifecycle updates, review feedback remediation, and final sign-off.
- Parent agent establishes the focused test contract before production
  implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  closes after planned production code changes are complete.
- Parent agent may update docs, integration tests, static checks, and evidence
  while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused tests established
  - Covers: implementation step 1.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3_bilibili.py tests\test_web_agent3_routing.py::test_web_agent3_router_normalizes_final_source_action_matrix -q`
  - Evidence: record expected failures or baseline results. Sign-off:
    `<agent/date>`.

- [ ] Stage 2 - Bilibili source module implemented
  - Covers: implementation steps 2-3.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3_bilibili.py -q`
  - Evidence: changed production files and test output. Sign-off:
    `<agent/date>`.

- [ ] Stage 3 - discovery, routing, and prompt roster integrated
  - Covers: implementation step 4.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3.py::test_web_agent3_source_subagents_are_discovered_from_subagent_package tests\test_web_agent3.py::test_web_agent3_router_prompt_lists_enabled_sources_only tests\test_web_agent3_routing.py -q`
  - Evidence: source roster and routing test output. Sign-off:
    `<agent/date>`.

- [ ] Stage 4 - docs and packaging complete
  - Covers: HOWTO, ICD, registry, and optional dependency declaration.
  - Verify: static greps in `Verification`.
  - Evidence: doc and `pyproject.toml` diffs. Sign-off: `<agent/date>`.

- [ ] Stage 5 - full verification complete
  - Covers: implementation step 5.
  - Verify: every command in `Verification`.
  - Evidence: command summaries and allowed grep matches. Sign-off:
    `<agent/date>`.

- [ ] Stage 6 - independent code review complete
  - Covers: implementation step 6.
  - Verify: rerun affected checks after fixes.
  - Evidence: findings, fixes, rerun commands, residual risks, and approval
    status. Sign-off: `<agent/date>`.

## Verification

### Static Compile

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py tests\test_web_agent3_bilibili.py tests\test_web_agent3.py tests\test_web_agent3_routing.py`
  - Expected: command exits 0.

### Tests

- `venv\Scripts\python.exe -m pytest tests\test_web_agent3_bilibili.py -q`
  - Expected: all tests pass.
- `venv\Scripts\python.exe -m pytest tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_bilibili.py -q`
  - Expected: all tests pass.

### Static Greps

- `rg -n "bilibili-api-python" pyproject.toml`
  - Expected: one optional dependency match under `[project.optional-dependencies]`.
- `rg -n "bilibili_api" src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py tests\test_web_agent3_bilibili.py`
  - Expected: matches only in Bilibili source module and Bilibili tests.
- `rg -n "^from bilibili_api|^import bilibili_api" src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py`
  - Expected: no matches. Lazy imports inside helper or execution functions
    are allowed when they are indented.
- `rg -ni "BILIBILI_TOKEN|BILIBILI_COOKIE|BILIBILI_SESSDATA|BILIBILI_CREDENTIAL|BILI_JCT|BUVID|DEDEUSERID" src docs tests pyproject.toml`
  - Expected: no matches. This plan adds no Bilibili environment secrets or
    account credential settings.
- `rg -n "youtube|reddit" src\kazusa_ai_chatbot\rag\web_agent3\subagent tests\test_web_agent3_bilibili.py`
  - Expected: no matches. This plan does not add other site sources.

### Dependency Availability Smoke

- `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.rag.web_agent3 import subagent; print(subagent._SUBAGENT_NAMES)"`
  - Expected: command exits 0 whether `bilibili_api` is installed or absent.

### Diff Hygiene

- `git diff --check`
  - Expected: command exits 0.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Alignment with this plan's read-only Bilibili source contract.
- Optional dependency declaration and import-time availability behavior.
- Absence of hidden fallback paths to `web_search`, `web_read`, browser
  automation, credentials, or account APIs.
- Bilibili API response compaction, prompt safety, result caps, and bounded
  error behavior.
- Router prompt roster and disabled-source normalization.
- Deterministic test quality, including fake dependency injection and no live
  external-service dependency in the regular suite.
- HOWTO and web_agent3 ICD accuracy.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `pyproject.toml` declares Bilibili as an optional dependency extra.
- `subagent/bilibili.py` exists and exposes the required source interface.
- `bilibili` registers only when `bilibili_api` is importable.
- Missing `bilibili_api` does not break web_agent3 import or source discovery.
- `bilibili` supports only `read` and `search`.
- Bilibili video URLs, BV ids, and AV ids are parsed deterministically.
- Bilibili read returns compact metadata and subtitle evidence when available.
- Bilibili read clearly marks metadata-only evidence when subtitles are absent.
- Bilibili search returns bounded video candidates ordered by popularity for
  hottest-video requests.
- Explicit disabled `bilibili` decisions normalize to `stop`.
- No Bilibili `.env` secret, credential, browser, account, or write action is
  added.
- Focused and regression tests pass.
- Independent code review is complete.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Missing optional dependency breaks import | Lazy imports and `is_enabled()` dependency probe | Dependency availability smoke |
| Router sees unavailable Bilibili source | Discovery registers enabled modules only | Source discovery tests |
| Disabled Bilibili silently falls back to generic web search | Explicit disabled-source normalization to `stop` | Routing tests |
| API response shape changes | Field-safe compaction with bounded errors | Fixture tests for partial/missing fields |
| "讲了什么内容" overclaims without subtitles | Evidence basis labels and limitation message | Read metadata-only test |
| Optional dependency bloats core install | Declare optional extra only | `pyproject.toml` grep |

## Execution Evidence

- No implementation evidence has been recorded. Status is `draft`.
