# web_agent3 bilibili source subagent plan

## Summary

- Goal: Add Bilibili as a real read-only public-content `web_agent3` source
  subagent backed by `bilibili-api-python`.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`
- Overall cutover strategy: bigbang for the Bilibili source. Add one final
  `bilibili` source module with no legacy placeholder fallback.
- Highest-risk areas: optional dependency discovery, import-time failure when
  the dependency is absent, prompt roster drift, disabled-source fallback to
  generic web search, Bilibili API response shape drift across content
  families, transport selection, subtitle-body retrieval, semantic search
  scope selection, and over-broad metadata leakage.
- Acceptance criteria: Bilibili registers only when `bilibili_api` is
  importable, supports only `read` and `search`, reads public Bilibili content
  URLs through source-local parsing, supports video BV/AV id reads, searches
  Bilibili content semantically with bounded result compaction, and preserves
  the existing `WebAgent3().run(...)` public contract.

## Context

`web_agent3` now discovers source modules from `subagent/*.py`. Available
sources expose `SOURCE`, `DESCRIPTION`, `SUPPORTED_ACTIONS`, and
`execute(decision)`. Optional sources expose `is_enabled()` and stay out of
the router prompt when unavailable.

The user wants Bilibili support for read-only evidence requests such as:

- `https://www.bilibili.com/video/BV1CqV266EJY/ 讲了什么内容`
- `帮我在bilibili上搜索关于vibe coding相关视频并且推荐给我一个最热门的视频`

Follow-up user decision:

- Bilibili search must be semantic, not a video-only API wrapper. The subagent
  honors an explicit content type when the user asks for one, and it chooses
  the Bilibili search scope internally when the user provides only a semantic
  topic.
- Bilibili read must be link-driven. The source module detects the public
  Bilibili URL family from the supplied link and dispatches to the matching
  read-only handler inside `subagent/bilibili.py`.

Research findings checked on 2026-07-02:

- `bilibili-api-python` is published on PyPI as version `17.4.2`, with
  project import package `bilibili_api`, Python requirement `>=3.10`, and
  installation command `pip install bilibili-api-python`.
- The installation reference for docs and implementation planning is
  `https://pypi.org/project/bilibili-api-python/`.
- PyPI describes the package as covering Bilibili videos, audio, live,
  dynamics, articles, users, bangumi, and related common APIs.
- `bilibili_api.video.Video` accepts `bvid` or `aid` and exposes
  `get_info()`, `get_pages()`, `get_related()`, and `get_subtitle(cid)`.
- `get_subtitle(cid)` returns subtitle metadata from player info. Subtitle
  body text requires fetching the bounded subtitle JSON URL returned by that
  metadata when a subtitle entry is present.
- `bilibili_api.search.search_by_type(...)` supports video search with
  `SearchObjectType.VIDEO` and video ordering through `OrderVideo`, including
  `OrderVideo.TOTALRANK` for comprehensive ranking and `OrderVideo.CLICK` for
  most-clicked results.
- `bilibili_api.search.search(...)` provides general Bilibili search when the
  user did not specify a content family.
- The upstream package expects an async HTTP client library to be available.
  This project already has `httpx` as a core dependency, so the Bilibili source
  should select the `httpx` client after lazy SDK import.
- The upstream README warns that the site API can change quickly; field-safe
  compaction and fixture-driven response-shape tests are required.
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
- Use parent-led execution. In this plan, `subagent` means a `web_agent3`
  source module, not a coding-agent worker.
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
- Declare `bilibili-api-python` as a project optional dependency.
- Do not add runtime lazy installation in this plan.
- Do not add `.env` configuration for Bilibili in this plan.
- Do not use `web_search` or `web_read` as a hidden Bilibili fallback path.
- Do not add live external-service tests to the regular deterministic suite.
- Control optional dependency availability in tests with monkeypatches, fake
  modules, or subprocess import probes instead of relying on the ambient
  developer environment.

## Must Do

- Add a `bilibili` optional dependency extra to `pyproject.toml`.
- Add `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py`.
- Register Bilibili only when `bilibili_api` is importable.
- Expose `SOURCE = "bilibili"` and `SUPPORTED_ACTIONS = ("read", "search")`.
- Add a Bilibili `DESCRIPTION` that gives source-local query generation rules
  for public Bilibili URL reads and semantic Bilibili searches.
- Parse public Bilibili URL families inside `bilibili.py`.
- Parse BV ids and AV ids as video read targets inside `bilibili.py`.
- Implement URL-family read dispatch inside `bilibili.py`.
- Define a source-local supported content-family handler table for public
  Bilibili URL reads.
- Implement video reads using `bilibili_api.video.Video`.
- Implement non-video public-content reads with the matching upstream
  `bilibili_api` module when that public URL family is supported by the
  installed SDK.
- Return bounded `unsupported` observations for Bilibili URL families that the
  installed SDK cannot read without account credentials or unstable private
  endpoints.
- Implement semantic search planning inside `bilibili.py`.
- Implement unspecified-scope search using `bilibili_api.search.search`.
- Implement explicit typed search using `bilibili_api.search.search_by_type`
  with the matching `SearchObjectType` when available.
- Select the SDK `httpx` client after lazy import so the source uses the
  project's existing async HTTP dependency.
- Use type-specific comprehensive ranking for ordinary typed search requests.
- Use type-specific popularity ranking when the user asks for the hottest or
  most popular content and the upstream API supports that order.
- Fetch bounded subtitle JSON bodies from subtitle URLs returned by
  `get_subtitle(cid)` when subtitle metadata is present.
- Provide deterministic `stats_summary`, `content_basis`, and
  `popularity_basis` strings so the finalizer receives interpreted evidence.
- Return bounded prompt-safe observations with compact metadata and source
  evidence.
- Update router normalization with an explicit no-fallback source set for
  source-specific providers such as `nhentai` and `bilibili`.
- Update tests, ICD, HOWTO, and confirm the plan registry row.

## Deferred

- Do not add Bilibili login, cookies, credential storage, or account-scoped
  API access.
- Do not fetch or expose video binary downloads, image binaries, danmaku raw
  streams, comments, likes, favorites, history, or account state.
- Do not treat account-gated, private, or unstable unsupported Bilibili URL
  families as successful reads.
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
| Runtime API scope | bigbang | Support read-only public-content metadata, text evidence, and semantic search evidence only. |
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

- `read`: public Bilibili content URL. BV ids and AV ids are accepted as
  video read targets.
- `search`: semantic Bilibili content search. Explicit user content-type
  requests are honored, and unspecified content type uses source-local scope
  selection.

Supported read content families are the public Bilibili URL families that can
be mapped to read-only upstream SDK APIs in the installed
`bilibili-api-python` release. The initial handler set must include video,
article/read, bangumi/episode/film, live room metadata, user/space metadata,
dynamic/opus, audio, and topic when the installed SDK exposes stable
read-only APIs for those families. The source returns a bounded `unsupported`
observation when a Bilibili URL family requires account credentials, browser
execution, private endpoints, or an SDK API absent from the installed release.

`bilibili` observations are compact evidence packets. They include
`content_type`, canonical URL when available, title/name, creator display
name when available, summary or text excerpt, publish or activity time when
available, deterministic statistics summaries, source-specific compact fields,
and an evidence-basis label. Video reads may include page metadata and
subtitle-derived text when available. They must include a limitation note when
full content evidence is unavailable.

`bilibili` observations must not include cookies, credentials, account data,
raw headers, raw API blobs, binary media, images, comments, favorites, history,
or download URLs.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Source name | Use `bilibili` | The source name is direct and model-readable. |
| Dependency | Add `bilibili-api-python>=17.4.2,<18` as optional extra | The dependency is source-specific, current PyPI stable is `17.4.2`, and a bounded major range matches the local dependency style. |
| Availability | Gate on `importlib.util.find_spec("bilibili_api")` | Missing optional dependency disables the source without importing the SDK. |
| Transport | Select SDK client `httpx` after lazy SDK import | The project already depends on `httpx`; this keeps the optional extra small and avoids environment-dependent transport selection. |
| Runtime install | Exclude lazy installation | The project does not currently have a lazy dependency installer pattern. |
| Config | Add no Bilibili `.env` config | Public read/search use cases require no user-specific secret. |
| Installation docs | Reference `https://pypi.org/project/bilibili-api-python/` and document the project optional extra | The PyPI page is the user-approved installation reference for package name, current version, and Python requirement. |
| Read operation | Dispatch by public Bilibili URL family inside `bilibili.py` | The outer router should not know Bilibili content types or API modules. |
| Video read operation | Use `Video(...).get_info()` and `get_pages()` | These are the library's source-specific video metadata APIs. |
| Subtitle evidence | Use `get_subtitle(cid)` and bounded subtitle JSON URL fetches when subtitle metadata is present | Subtitle text is stronger evidence for "讲了什么内容" than metadata alone, while metadata-only reads must be labeled. |
| Search operation | Use general search for unspecified content scope and typed search for explicit content scope | The Bilibili subagent owns semantic scope selection; the outer router only selects `source=bilibili` and `action=search`. |
| Popularity ranking | Use type-specific popularity order only when the selected upstream search API supports it | Popularity requests should be grounded in an available provider order instead of invented ranking. |
| Disabled explicit source | Normalize explicit no-fallback source requests such as `nhentai` and `bilibili` to `stop` when the source is unavailable | A source-specific request should not silently become generic web search. |
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
  - Public Bilibili content URL
  - BV id as video shorthand
  - AV id as video shorthand

search query:
  - Bilibili semantic search need
  - Optional user language such as "视频", "专栏", "直播", "番剧",
    "UP 主", or "最热门" is handled by source-local scope and order
    selection.
```

Read observation shape:

```python
{
    "status": "success" | "partial" | "error",
    "source": "bilibili",
    "action": "read",
    "query": str,
    "target": {
        "content_type": str,
        "bvid": str | None,
        "aid": int | None,
        "public_id": str | int | None,
        "url": str | None,
        "canonical_url": str | None,
    },
    "content": {
        "content_type": str,
        "title": str,
        "creator": str | None,
        "summary": str,
        "published_or_active_at": str | None,
        "stats_summary": list[str],
        "content_excerpt": str,
        "content_basis": str,
        "content_fields": dict[str, str | int | list[str]],
        "limitation": str,
    },
    "message": str,
}
```

Video read observations may include `pages`, `duration`, `subtitle_excerpt`,
and `subtitle_basis` in `content_fields`.

Search observation shape:

```python
{
    "status": "success" | "partial" | "error",
    "source": "bilibili",
    "action": "search",
    "query": str,
    "content_scope": "general" | str,
    "order": "provider_default" | "comprehensive" | "popular" | str,
    "popularity_basis": str,
    "results": [
        {
            "rank": int,
            "content_type": str,
            "title": str,
            "url": str | None,
            "public_id": str | int | None,
            "creator": str | None,
            "published_or_active_at": str | None,
            "stats_summary": list[str],
            "summary": str,
            "recommendation_basis": str,
        }
    ],
    "message": str,
}
```

Failure conditions:

- Missing optional dependency: source is not registered.
- Unsupported action: return bounded `error` observation.
- Missing or unsupported Bilibili target for `read`: return bounded `error`
  or `unsupported` observation.
- API exception or response mismatch: return bounded `error` observation.
- Subtitle metadata URL fetch exception or response mismatch: continue with
  metadata-only evidence and a limitation message.
- Missing subtitle: return `partial` or `success` with
  `content_basis="metadata_only"` and a limitation message.
- Unknown typed search scope: use general search with
  `content_scope="general"` and a message describing the fallback.
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

No additional LLM call is added inside `subagent/bilibili.py`. Source-local
semantic search planning is deterministic: it preserves the user's semantic
search phrase, detects explicit content-scope requests from the selected
source query, chooses general search when scope is unspecified, and validates
provider-supported typed scopes before execution.

Context inputs:

- System prompt changes only in the generated source roster under source
  descriptions.
- Human payload remains unchanged.
- Bilibili source description must stay compact and process-stable for the
  Python session.
- Conservative static prompt growth is under 2,500 characters, below the
  default 50k-token planning cap.
- Bilibili observations must cap search results, content excerpts, subtitle
  excerpts, and source-specific compact fields so the evaluator/finalizer
  context remains bounded.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py`
  - Real Bilibili source subagent.
- `tests/test_web_agent3_bilibili.py`
  - Deterministic tests for Bilibili URL-family parsing, availability,
    read/search execution, semantic scope selection, and prompt-safe
    observation compaction.

### Modify

- `pyproject.toml`
  - Add optional dependency extra for Bilibili.
- `development_plans/README.md`
  - Confirm this active draft plan remains listed in the registry.
- `src/kazusa_ai_chatbot/rag/web_agent3/contracts.py`
  - Add a no-fallback source set for source-specific providers and stop
    explicit disabled `bilibili` router decisions instead of falling back to
    generic sources.
- `src/kazusa_ai_chatbot/rag/web_agent3/README.md`
  - Document the Bilibili source and optional dependency behavior.
- `docs/HOWTO.md`
  - Document installation command for Bilibili optional source and reference
    `https://pypi.org/project/bilibili-api-python/`.
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

- Actual problem: `web_agent3` cannot answer Bilibili-specific public-content
  read and search requests with source-specific API evidence.
- Minimal change: add one dependency-gated `bilibili` source module, one
  optional dependency extra, focused deterministic tests, and docs.
- Ownership boundaries: the router selects `bilibili`; `bilibili.py` owns
  Bilibili URL-family detection, semantic search scope selection,
  API/compaction; deterministic tests own API-shape fixtures; the existing
  finalizer owns visible answer synthesis.
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
     - `test_bilibili_is_disabled_when_dependency_probe_is_absent`
     - `test_bilibili_is_enabled_when_dependency_probe_is_present`
     - `test_bilibili_extracts_video_bv_and_av_targets`
     - `test_bilibili_supported_content_family_table_includes_public_families`
     - `test_bilibili_detects_public_url_families`
     - `test_bilibili_read_routes_article_url_to_article_handler`
     - `test_bilibili_read_routes_live_url_to_live_handler`
     - `test_bilibili_read_routes_space_url_to_user_handler`
     - `test_bilibili_read_reports_unsupported_url_family`
     - `test_bilibili_lazy_import_selects_httpx_client`
     - `test_bilibili_read_returns_compact_metadata_and_content_basis`
     - `test_bilibili_read_fetches_bounded_subtitle_json`
     - `test_bilibili_read_reports_metadata_only_when_subtitle_missing`
     - `test_bilibili_search_uses_video_click_order_for_hot_request`
     - `test_bilibili_search_uses_totalrank_order_for_ordinary_request`
     - `test_bilibili_search_uses_general_search_when_scope_unspecified`
     - `test_bilibili_search_uses_typed_search_when_scope_is_explicit`
     - `test_bilibili_search_returns_bounded_prompt_safe_mixed_candidates`
     - `test_bilibili_api_errors_are_bounded`
   - Update `tests/test_web_agent3.py` source discovery subprocess helper to
     support a fake `bilibili_api` package on `PYTHONPATH` and a controlled
     import-probe override for absent dependency cases.
   - Update `tests/test_web_agent3_routing.py` with enabled and disabled
     `bilibili` normalization cases.
   - Expected before implementation: tests fail because `bilibili.py` and the
     optional dependency extra do not exist.

2. Parent implements packaging and Bilibili module.
   - Add `[project.optional-dependencies].bilibili`.
   - Create `subagent/bilibili.py`.
   - Keep SDK imports lazy.
   - Implement availability, public URL-family detection, video shorthand
     parsing, read dispatch, semantic search scope selection, bounded errors,
     and prompt-safe compaction.
   - Update router normalization for explicit disabled `bilibili`.

3. Parent updates docs and integration tests.
   - Update web_agent3 ICD and HOWTO.
   - HOWTO installation guidance must reference
     `https://pypi.org/project/bilibili-api-python/`.
   - Update discovery and router prompt tests for dependency-gated Bilibili.
   - Run focused Bilibili and source discovery tests.

4. Parent runs full verification gates.
   - Run static compile, focused tests, web_agent3 regression tests, and static
     greps.
   - Record command summaries in `Execution Evidence`.

5. Parent performs an independent read-only code-review pass.
   - Review the implementation against this plan, source ownership, dependency
     gating, prompt safety, test determinism, and docs.
   - Fix approved-scope findings and rerun affected checks.

## Execution Model

- Parent agent owns orchestration, test code, production code, docs,
  verification, execution evidence, lifecycle updates, review feedback
  remediation, and final sign-off.
- Parent agent establishes the focused test contract before production
  implementation starts.
- Parent agent performs one independent read-only review pass after planned
  verification passes.
- The review pass reads the plan, diff, and evidence from a review stance,
  records findings, fixes approved-scope issues, and reruns affected checks.

## Progress Checklist

- [ ] Stage 1 - focused tests established
  - Covers: implementation step 1.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3_bilibili.py tests\test_web_agent3_routing.py::test_web_agent3_router_normalizes_final_source_action_matrix -q`
  - Evidence: record expected failures or baseline results. Sign-off:
    `<agent/date>`.

- [ ] Stage 2 - Bilibili source module implemented
  - Covers: implementation step 2.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3_bilibili.py -q`
  - Evidence: changed production files and test output. Sign-off:
    `<agent/date>`.

- [ ] Stage 3 - discovery, routing, and prompt roster integrated
  - Covers: implementation step 3.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3.py::test_web_agent3_source_subagents_are_discovered_from_subagent_package tests\test_web_agent3.py::test_web_agent3_router_prompt_lists_enabled_sources_only tests\test_web_agent3_routing.py -q`
  - Evidence: source roster and routing test output. Sign-off:
    `<agent/date>`.

- [ ] Stage 4 - docs and packaging complete
  - Covers: HOWTO, ICD, registry, and optional dependency declaration.
  - Verify: static greps in `Verification`.
  - Evidence: doc and `pyproject.toml` diffs. Sign-off: `<agent/date>`.

- [ ] Stage 5 - full verification complete
  - Covers: implementation step 4.
  - Verify: every command in `Verification`.
  - Evidence: command summaries and allowed grep matches. Sign-off:
    `<agent/date>`.

- [ ] Stage 6 - independent code review complete
  - Covers: implementation step 5.
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

- `rg -n "bilibili-api-python>=17\\.4\\.2,<18" pyproject.toml`
  - Expected: one optional dependency match under `[project.optional-dependencies].bilibili`.
- `rg -n "https://pypi.org/project/bilibili-api-python/" docs\HOWTO.md src\kazusa_ai_chatbot\rag\web_agent3\README.md`
  - Expected: at least one documentation match. Installation guidance must
    cite the user-approved PyPI project page.
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

## Independent Plan Review

Plan review performed on 2026-07-02 against the current web_agent3 ICD,
current routing/discovery tests, current `pyproject.toml`, PyPI metadata, and
upstream `bilibili-api-python` source files.

| Severity | Finding | Resolution |
|---|---|---|
| High | The plan depended on native coding-agent subagents even though the change is deterministic and scoped. | Execution model is now parent-led; independent review is a separate read-only pass. |
| High | Optional dependency tests could pass or fail based on the ambient developer environment. | Tests must control dependency probes with monkeypatches, fake modules, or subprocess overrides. |
| High | Disabled-source routing was phrased as a Bilibili-only special case while current routing already has source-specific no-fallback behavior for nHentai. | Contracts now require a shared no-fallback source set for `nhentai` and `bilibili`. |
| High | Subtitle evidence assumed `get_subtitle(cid)` directly supplied body text. | Plan now requires bounded subtitle JSON URL fetches when metadata is present and metadata-only degradation when fetches fail. |
| Medium | Raw stat dictionaries and raw view counts left semantic interpretation to the local LLM. | Observation contracts now require deterministic `stats_summary`, `content_basis`, and `popularity_basis` strings. |
| Medium | Video search order was always `OrderVideo.CLICK`, which would distort ordinary relevance searches. | Typed video searches use `OrderVideo.TOTALRANK` by default and `OrderVideo.CLICK` for hot/popular requests; unspecified-scope searches use general provider search. |
| Medium | Transport selection was left to the upstream SDK environment. | Plan now selects the SDK `httpx` client after lazy import, using the project's existing dependency. |
| Low | Plan registry action said to add a row even though the current registry already lists the draft plan. | Action now says to confirm the existing active draft registry row. |

Follow-up plan update on 2026-07-02:

| Severity | Finding | Resolution |
|---|---|---|
| High | The draft scope was still video-only while the intended Bilibili source must support non-video content. | The plan now defines Bilibili as a public-content source with URL-family read dispatch and semantic search. |
| High | Search scope was too closely tied to `SearchObjectType.VIDEO`. | The plan now uses general search for unspecified scope and typed search only when the user explicitly asks for a content family. |
| Medium | Installation guidance did not explicitly cite the user-approved package reference. | HOWTO and ICD verification now require `https://pypi.org/project/bilibili-api-python/`. |

Plan review status: addressed in this draft. Production implementation remains
unstarted.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must perform this as a separate read-only review pass after
implementation and verification evidence exists.

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
- Public Bilibili URL families are parsed deterministically.
- BV ids and AV ids are parsed deterministically as video read targets.
- Bilibili read returns compact metadata, text evidence, and content-family
  labels when available.
- Video reads return subtitle evidence when available.
- Video reads clearly mark metadata-only evidence when subtitles are absent.
- Subtitle body fetches are bounded and degrade to metadata-only evidence on
  fetch or response-shape failure.
- Bilibili observations include deterministic `stats_summary`,
  `content_basis`, and `popularity_basis` strings where applicable.
- Bilibili search uses general provider search when the user gives no content
  family.
- Bilibili search uses typed provider search when the user explicitly names a
  supported content family.
- Bilibili search returns bounded mixed or typed candidates with content-family
  labels.
- Bilibili typed popularity requests use provider-supported popularity order
  when available.
- Bilibili installation docs reference
  `https://pypi.org/project/bilibili-api-python/`.
- Explicit disabled `bilibili` decisions normalize to `stop`.
- No Bilibili `.env` secret, credential, browser, account, or write action is
  added.
- Focused and regression tests pass.
- Independent code review is complete.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Missing optional dependency breaks import | Lazy imports and `is_enabled()` dependency probe | Dependency availability smoke |
| Ambient installed package changes discovery tests | Controlled dependency probes and fake modules | Source discovery tests |
| Router sees unavailable Bilibili source | Discovery registers enabled modules only | Source discovery tests |
| Disabled Bilibili silently falls back to generic web search | No-fallback source normalization to `stop` | Routing tests |
| API response shape changes | Field-safe compaction with bounded errors | Fixture tests for partial/missing fields |
| Public content-family APIs differ by SDK release | URL-family handler table and unsupported observations | URL-family read tests |
| Search scope selection becomes hidden routing in the outer graph | Keep scope selection inside `bilibili.py` and pass only `action`, `source`, `query` through the router | Router prompt and source execution tests |
| "讲了什么内容" overclaims without subtitles | Content basis labels and limitation message | Read metadata-only test |
| Upstream transport selection drifts by local environment | Select SDK `httpx` client after lazy import | Focused Bilibili transport test |
| Optional dependency bloats core install | Declare optional extra only | `pyproject.toml` grep |

## Execution Evidence

- No implementation evidence has been recorded. Status is `draft`.
