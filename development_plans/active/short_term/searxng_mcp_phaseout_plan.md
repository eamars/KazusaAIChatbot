# searxng mcp phaseout plan

## Summary

- Goal: Replace `web_agent3` generic SearXNG MCP search/read calls with direct
  process-owned web search and URL access while preserving the generic MCP
  adapter for non-SearXNG tools.
- Plan class: large
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `development-plan`.
- Overall cutover strategy: bigbang.
- Highest-risk areas: direct URL-read behavior, local resource access,
  SearXNG absence handling, prompt-facing evidence quality, and stale MCP
  references in tests/docs.
- Acceptance criteria: `web_agent3` contains no `mcp-searxng` tool calls,
  URL reads work without SearXNG, search degrades cleanly when `SEARXNG_URL` is
  absent, generic MCP support remains intact, and focused/regression checks pass.

## Context

The current main-branch `web_agent3` generic source calls
`mcp-searxng__searxng_web_search` and `mcp-searxng__web_url_read` through
`mcp_manager` in `src/kazusa_ai_chatbot/rag/web_agent3/searxng_tools.py`.
This creates a required SearXNG MCP service even though SearXNG search can be
called directly through its JSON search API and URL reading is not a SearXNG
API feature at all.

The user has confirmed these decisions:

- Use a bigbang cutover. Do not keep SearXNG MCP as a compatibility path.
- Preserve the generic MCP adapter capability for unrelated MCP servers.
- Strip SearXNG-specific MCP names and setup from runtime code, live tests, and
  operator docs.
- Read all new web facility settings from `kazusa_ai_chatbot.config`.
- Do not hard-code `localhost`, `192.168.2.10`, or any other SearXNG endpoint
  into runtime code or tests.
- `SEARXNG_URL` may be absent. Its absence must not crash config import,
  service startup, or `web_agent3`; it only makes search unavailable.
- URL access is available by default and belongs to the Kazusa process, not to
  SearXNG.
- URL access must allow local resources such as `localhost`, loopback, private
  LAN addresses, and intranet hostnames.
- URL access must use a basic browser-compatible request profile, including a
  configurable browser-like User-Agent.
- Execution must stay on branch `searxng-mcp-phaseout`. Do not merge,
  cherry-pick, or otherwise integrate this work into `main` during this plan.
- The current SearXNG MCP baseline must be recorded with 10 examples before
  production-code changes, and sign-off must prove the direct workflow has no
  correctness or material latency regressions against that baseline.

External API fact used by this plan:

- SearXNG search uses the documented `/search` endpoint with `format=json`.
- The current MCP server's `web_url_read` behavior is implemented by the MCP
  server as direct URL fetch plus HTML conversion. It is not a SearXNG endpoint.

References:

- SearXNG Search API:
  https://docs.searxng.org/dev/search_api.html
- Current SearXNG MCP server:
  https://github.com/ihor-sokoliuk/mcp-searxng
- SearXNG MCP search implementation:
  https://github.com/ihor-sokoliuk/mcp-searxng/blob/main/src/search.ts
- SearXNG MCP URL reader implementation:
  https://github.com/ihor-sokoliuk/mcp-searxng/blob/main/src/url-reader.ts

## Mandatory Skills

- `py-style`: load before editing Python production files or Python tests.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing `web_agent3`, RAG prompts,
  RAG routing, evidence shape, or LLM-facing observations.
- `development-plan`: load before executing, reviewing, updating lifecycle
  status, or signing off this plan.

## Mandatory Rules

- Do not read `.env`.
- Use `venv\Scripts\python` for Python and pytest commands.
- Use `apply_patch` for manual edits.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Do not execute this plan while its status is `draft`.
- Do not modify production code unless the user explicitly commands
  implementation after approving this plan.
- Execute this plan only on branch `searxng-mcp-phaseout`.
- Do not merge, rebase into, cherry-pick into, or otherwise integrate this
  branch to `main` as part of this plan. Final sign-off stops with the feature
  branch ready for user review.
- Keep `MCP_SERVERS`, `MCP_CALL_TIMEOUT`, `MCP_CONNECT_TIMEOUT`,
  `mcp_client.py`, service MCP startup/shutdown, and generic MCP tests unless
  a reference is SearXNG-specific.
- Do not add a SearXNG MCP fallback, compatibility shim, dual path, feature
  flag, or legacy adapter mode.
- Do not add URL resource denylisting for localhost, loopback, RFC1918 private
  ranges, or intranet hostnames.
- Do not add `file://`, shell, filesystem, browser automation, cookies,
  persistent sessions, rotating proxies, CAPTCHA handling, or credential
  forwarding.
- Do not move web action selection into deterministic code. The LLM router
  still chooses only `search`, `read`, or `stop`; deterministic code owns
  execution, validation, limits, and formatting.
- Runtime code must read SearXNG and URL-reader settings from
  `kazusa_ai_chatbot.config`, not from ad hoc `os.getenv` calls outside
  `config.py`.
- The existing required LLM and embedding environment variables in `config.py`
  remain required. The new guarantee is that absent `SEARXNG_URL` creates no
  additional import/startup crash and only disables search execution.

## Must Do

- Replace `web_agent3` generic search execution with direct SearXNG JSON API
  calls.
- Replace `web_agent3` generic URL-read execution with process-local HTTP(S)
  URL reads.
- Keep URL-read available by default even when `SEARXNG_URL` is absent.
- Make direct SearXNG search unavailable, not crashing, when `SEARXNG_URL` is
  absent or blank.
- Add `config.py` settings for direct search and URL-reader behavior.
- Update focused deterministic tests for direct search, direct URL reads,
  config import behavior, local URL allowance, user-agent header behavior, and
  retained generic MCP support.
- Update live web-agent test setup so it checks `SEARXNG_URL` for search
  availability instead of MCP tool discovery.
- Update README/HOWTO/web-agent ICD docs to remove SearXNG MCP as the web
  setup path and document direct `SEARXNG_URL` plus optional generic MCP
  servers.
- Update MCP tests to avoid SearXNG-specific server/tool names when testing the
  generic MCP adapter.
- Preserve and compare a 10-example SearXNG MCP baseline before implementation
  and a 10-example direct-workflow regression artifact before sign-off.
- Address any failed direct-workflow baseline comparison before sign-off unless
  the user explicitly accepts the regression.
- Run every verification gate listed in this plan.

## Deferred

- Do not add source-specific Bilibili or YouTube provider APIs.
- Do not redesign the `web_agent3` LLM router, evaluator, finalizer, or public
  helper contract.
- Do not change RAG initializer slot policy, RAG dispatcher policy, projection
  fields, Cache2 policy, cognition prompts, dialog prompts, consolidation, or
  adapters.
- Do not add persistent caching for web search or URL reads.
- Do not add new third-party HTML extraction dependencies unless the standard
  library and existing `httpx` cannot satisfy the focused tests.
- Do not remove the generic MCP subsystem.
- Do not add a new service process or local proxy.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| `web_agent3` generic search | bigbang | Replace SearXNG MCP tool calls with direct SearXNG JSON API calls. No MCP fallback. |
| `web_agent3` generic URL read | bigbang | Replace SearXNG MCP URL reads with process-local HTTP(S) reads. No MCP fallback. |
| Generic MCP adapter | compatible | Preserve generic MCP support for non-SearXNG tools only. Remove SearXNG-specific names from generic MCP tests/docs. |
| Config | bigbang | Add direct web facility config in `config.py`; runtime code imports constants from config. |
| Tests/docs | bigbang | Rewrite SearXNG MCP expectations into direct SearXNG and direct URL-reader expectations. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- In bigbang areas, delete or rewrite legacy references instead of preserving
  them.
- In compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The completed web evidence path is:

```text
web_agent3 router decision
  -> source subagent
  -> search: direct SearXNG /search?format=json through config.SEARCH settings
  -> read: direct process-local HTTP(S) URL reader through config.URL settings
  -> evaluator/finalizer unchanged
  -> WebAgent3.run unchanged public result shape
```

If `SEARXNG_URL` is absent or blank, `web_search(...)` returns a bounded
prompt-facing unavailable observation. It does not raise during config import,
service startup, package import, or normal `web_agent3` graph execution.

`web_url_read(...)` remains available by default and can access local resources
reachable from the Kazusa process. It sends a configurable browser-like
User-Agent and bounded browser-compatible headers. It accepts only `http` and
`https` URLs. It follows redirects within a fixed cap, applies timeouts and
response-size limits, and extracts text from HTML, plain text, JSON, and
XML-like content. It supports the current `startChar`, `maxLength`, `section`,
`paragraphRange`, and `readHeadings` tool arguments with the exact behavior in
`Contracts And Data Shapes`.

The public `WebAgent3.run(...)` contract remains:

```python
{
    "resolved": bool,
    "result": str,
    "attempts": int,
    "cache": dict,
}
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Search transport | Direct SearXNG JSON API | SearXNG search is already available as HTTP JSON and does not need MCP. |
| URL-read transport | Direct process HTTP(S) read | URL reading is not a SearXNG API. Process-local reads preserve local resource perspective. |
| SearXNG absence | Search returns unavailable observation | Missing search config should not crash the service or block URL reads. |
| URL local resources | Allow local/private/intranet HTTP(S) URLs | User explicitly requires local resource access. |
| Basic anti-robot behavior | Static configurable browser-like User-Agent and request headers | Covers common bot-client rejection without unbounded evasion mechanisms. |
| MCP subsystem | Preserve generic MCP, remove SearXNG-specific usage | User wants MCP adapter capability retained while phasing out SearXNG MCP. |
| Config ownership | `config.py` constants only | Matches project configuration style and avoids hidden defaults at call sites. |

## Contracts And Data Shapes

Add these config constants in `src/kazusa_ai_chatbot/config.py`:

```python
SEARXNG_URL: str
SEARXNG_SEARCH_TIMEOUT_SECONDS: float
SEARXNG_SEARCH_RESULT_LIMIT: int
WEB_URL_READ_TIMEOUT_SECONDS: float
WEB_URL_READ_MAX_BYTES: int
WEB_URL_READ_MAX_CHARS: int
WEB_URL_READ_REDIRECT_LIMIT: int
WEB_URL_READER_USER_AGENT: str
WEB_URL_READER_ACCEPT_LANGUAGE: str
```

Required config behavior:

- `SEARXNG_URL` reads from environment with default `""`, strips surrounding
  whitespace, strips trailing slashes, and accepts either empty string,
  `http://...`, or `https://...`.
- Empty `SEARXNG_URL` is valid config and means search is unavailable.
- If non-empty `SEARXNG_URL` does not start with `http://` or `https://`,
  config import raises `ValueError("SEARXNG_URL must be empty or an HTTP(S) URL")`.
- Existing required LLM and embedding environment variables remain required.
  Tests may populate placeholder values for those unrelated required settings,
  but production code must not make them optional for this plan.
- Timeout, byte, char, result, and redirect settings are validated through
  config helper functions and fail fast only when explicitly invalid.
- Runtime web modules import these constants from `config.py`.

Config defaults and bounds:

| Constant | Environment variable | Default | Bounds or validation |
|---|---|---|---|
| `SEARXNG_URL` | `SEARXNG_URL` | `""` | Empty or HTTP(S) URL after stripping whitespace/trailing slashes |
| `SEARXNG_SEARCH_TIMEOUT_SECONDS` | `SEARXNG_SEARCH_TIMEOUT_SECONDS` | `30` | Float between `1.0` and `120.0` inclusive |
| `SEARXNG_SEARCH_RESULT_LIMIT` | `SEARXNG_SEARCH_RESULT_LIMIT` | `10` | Integer between `1` and `20` inclusive |
| `WEB_URL_READ_TIMEOUT_SECONDS` | `WEB_URL_READ_TIMEOUT_SECONDS` | `30` | Float between `1.0` and `120.0` inclusive |
| `WEB_URL_READ_MAX_BYTES` | `WEB_URL_READ_MAX_BYTES` | `1048576` | Integer between `1024` and `5242880` inclusive |
| `WEB_URL_READ_MAX_CHARS` | `WEB_URL_READ_MAX_CHARS` | `10000` | Integer between `1000` and `50000` inclusive |
| `WEB_URL_READ_REDIRECT_LIMIT` | `WEB_URL_READ_REDIRECT_LIMIT` | `5` | Integer between `0` and `10` inclusive |
| `WEB_URL_READER_USER_AGENT` | `WEB_URL_READER_USER_AGENT` | `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36` | Non-empty after stripping whitespace |
| `WEB_URL_READER_ACCEPT_LANGUAGE` | `WEB_URL_READER_ACCEPT_LANGUAGE` | `en-US,en;q=0.9` | Non-empty after stripping whitespace |

Direct search contract:

```python
@tool
async def web_search(
    query: str,
    pageno: int = 1,
    time_range: str = "",
    language: str = "",
) -> str:
```

- When configured, call `{SEARXNG_URL}/search` with `q`, `format=json`,
  `pageno`, and `safesearch=0`; include `time_range` and `language` only when
  non-empty after stripping whitespace, because the observed SearXNG instance
  rejects blank optional parameters with HTTP 400.
- Parse `results[]` rows into bounded text records with title, URL, snippet,
  engine/source, and score when present.
- When no search URL is configured, return:
  `Error: SearXNG search unavailable: SEARXNG_URL is not configured.`
- On network, timeout, HTTP, or JSON errors, return a bounded `Error:` string
  rather than raising through the graph.

Direct URL-read contract:

```python
@tool
async def web_url_read(
    url: str,
    startChar: int = 0,
    maxLength: int = 10000,
    section: str = "",
    paragraphRange: str = "",
    readHeadings: bool = False,
) -> str:
```

- Accept only `http` and `https` URLs.
- Allow local/private/intranet targets.
- Send configured `WEB_URL_READER_USER_AGENT` and
  `WEB_URL_READER_ACCEPT_LANGUAGE`.
- Apply config-owned timeout, redirect, byte, and char caps.
- If `maxLength <= 0`, use config-owned max char cap rather than unbounded
  output.
- Convert HTML into readable text using a deterministic local extractor.
- For HTML, extract headings from `h1` through `h6`, paragraph/list/table cell
  text, and visible title text while ignoring `script`, `style`, `noscript`,
  `template`, `svg`, and metadata content.
- For non-HTML textual responses, decode as UTF-8 with replacement and return
  bounded text without HTML-specific heading extraction.
- For `readHeadings=True`, return one heading per line for HTML; for non-HTML
  text with no headings, return `Error: no headings found`.
- For `section`, return content after the first heading whose text contains
  the section query case-insensitively, stopping before the next heading at the
  same or higher level. If no heading matches, return
  `Error: section not found: <section>`.
- For `paragraphRange`, support `N`, `N-M`, and `N-` paragraph selection after
  extraction using one-based paragraph indexes. Invalid ranges return
  `Error: invalid paragraphRange: <paragraphRange>`.
- Apply paragraph selection before `startChar` and `maxLength` slicing.
- Apply `startChar` as a zero-based character offset after extraction and
  paragraph/section filtering. Negative `startChar` is treated as `0`.
- Apply `maxLength` after `startChar`; values greater than
  `WEB_URL_READ_MAX_CHARS` are capped to `WEB_URL_READ_MAX_CHARS`.
- Return bounded `Error:` strings for invalid URLs, unsupported schemes,
  network failures, timeout, over-limit, empty content, and unsupported binary
  content.

Baseline and regression artifact contract:

```python
{
    "summary": {
        "started_at": str,
        "branch": str,
        "baseline": str,
        "mcp_url": str,
        "searxng_url_observed": str,
        "connect_ms": float,
        "tools": list[str],
        "example_count": int,
        "success_count": int,
        "failure_count": int,
        "latency_ms_min": float,
        "latency_ms_median": float,
        "latency_ms_max": float,
        "successful_latency_ms_median": float,
    },
    "examples": [
        {
            "id": str,
            "kind": "search" | "read",
            "tool": str,
            "arguments": dict,
            "success": bool,
            "latency_ms": float,
            "output": {
                "chars": int,
                "sha256": str,
                "first_line": str,
                "excerpt": str,
            },
        }
    ],
}
```

The baseline artifact is
`development_plans/active/short_term/artifacts/searxng_mcp_baseline_2026-06-01.json`.
The direct-workflow comparison artifact must be
`development_plans/active/short_term/artifacts/searxng_direct_regression_2026-06-01.json`.
For the direct artifact, keep the same summary schema and set `mcp_url` to
`""` because MCP is not used by the direct workflow.

Baseline cases:

| Case | Kind | Input |
|---|---|---|
| `search_01` | search | `SearXNG search API format json` |
| `search_02` | search | `Python httpx AsyncClient timeout documentation` |
| `search_03` | search | `OpenAI API responses documentation` |
| `search_04` | search | `Kazusa AI chatbot github` |
| `search_05` | search | `Wikipedia Kyoto population` |
| `read_01` | read | `https://example.com/` |
| `read_02` | read | `https://docs.searxng.org/dev/search_api.html` |
| `read_03` | read | `https://httpbin.org/html` |
| `read_04` | read | `http://192.168.2.10:8080/` |
| `read_05` | read | `http://192.168.2.10:8080/search?q=kazusa%20test&format=json` |

Direct workflow regression criteria:

- All 10 direct cases must return `success=true`.
- Search cases must return at least one title/URL record or a non-empty
  no-results record from SearXNG. They must not return MCP-related errors.
- `read_01` must include `Example Domain`.
- `read_02` must include `Search API`.
- `read_03` must include `Herman Melville`.
- `read_04` must return non-empty local SearXNG page text.
- `read_05` must include `kazusa test` or `"query"`.
- Direct median latency must be less than or equal to
  `baseline.successful_latency_ms_median * 1.25`.
- Each direct case latency must be less than or equal to the larger of
  `baseline case latency * 2.0` and `baseline case latency + 1500ms`.
- If a comparison fails, fix the direct workflow and rerun the comparison
  before sign-off. Do not classify a failed comparison as accepted unless the
  user explicitly accepts that regression.

## LLM Call And Context Budget

Before this plan:

- `web_agent3` uses one router/generator LLM call, one deterministic MCP tool
  call, one evaluator LLM call per loop, and one finalizer LLM call.
- `MAX_WEB_SEARCH_AGENT_RETRY` caps loop iterations.

After this plan:

- LLM call count is unchanged.
- Router/generator, evaluator, and finalizer prompt contracts remain unchanged
  except for source description wording that removes SearXNG MCP transport
  details and documents local URL reads semantically.
- Tool observations remain bounded text/JSON-serializable records.
- Direct search and URL read replace MCP latency with direct `httpx` latency
  bounded by config timeouts.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/web_agent3/direct_searxng.py`: direct SearXNG JSON
  search client and result formatter.
- `src/kazusa_ai_chatbot/rag/web_agent3/url_reader.py`: direct process-local
  HTTP(S) URL reader, text extraction, heading/section/paragraph/char slicing,
  browser-like request headers, and bounded error strings.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add direct search and URL-reader config
  constants and validation.
- `src/kazusa_ai_chatbot/rag/web_agent3/searxng_tools.py`: keep public tool
  names but replace MCP calls with direct modules.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/generic.py`: update docstring
  and source description to remove SearXNG MCP wording while keeping the same
  `execute(...)` contract.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py` and
  `src/kazusa_ai_chatbot/rag/web_agent3/subagent/youtube.py`: update wording
  only if current comments or log messages imply SearXNG MCP fallback.
- `src/kazusa_ai_chatbot/rag/web_agent3/README.md`: document direct SearXNG
  search, direct process URL read, local resource allowance, and retained
  generic MCP boundary.
- `src/kazusa_ai_chatbot/rag/README.md`: update external retrieval operational
  note if it mentions MCP-backed web.
- `README.md`: replace "optional MCP web tools" wording with direct web
  facility plus optional generic MCP tools.
- `docs/HOWTO.md`: remove `mcp-searxng` setup as the web-search requirement;
  add direct `SEARXNG_URL` and URL-reader config examples; keep `MCP_SERVERS`
  documented for optional non-SearXNG MCP tools.
- `tests/test_config.py`: add optional `SEARXNG_URL` and URL-reader config
  import/validation tests.
- `tests/test_web_agent3.py`: replace MCP delegation tests with direct search
  and URL-reader tests.
- `tests/test_web_agent3_routing.py`: update monkeypatches and wording while
  preserving routing contract tests.
- `tests/test_e2e_live_llm.py`: update live web-agent prerequisite from MCP
  tool discovery to `SEARXNG_URL` presence for search-backed live cases.
- `tests/test_mcp_client.py`: preserve generic MCP behavior tests but remove
  SearXNG-specific server/tool names from examples.

### Delete

- Delete no source file in this plan.
- Delete SearXNG MCP references from active runtime docs/tests/source where
  they describe current web search behavior.

### Keep

- `src/kazusa_ai_chatbot/mcp_client.py`: keep generic MCP manager unchanged
  unless a SearXNG-specific example or comment must be made generic.
- Service startup/shutdown MCP lifecycle in `src/kazusa_ai_chatbot/service.py`.
- Existing `web_agent3` public helper contract, source subagent discovery,
  router decision shape, evaluator/finalizer flow, and nHentai provider.
- Archived development plans may retain historical SearXNG MCP references.

## Overdesign Guardrail

- Actual problem: generic web search/read currently requires a SearXNG MCP
  service even though direct SearXNG search and process-local URL reads are the
  desired runtime boundary.
- Minimal change: replace only the generic `web_search` and `web_url_read`
  tool bodies and their tests/docs while preserving `web_agent3` router,
  evaluator, finalizer, source dispatch, RAG projection, and generic MCP
  manager.
- Ownership boundaries: the LLM router chooses semantic action/source/query;
  deterministic `web_agent3` code validates config, executes direct HTTP,
  enforces limits, and formats observations; RAG/cognition/dialog consume the
  same evidence shape as before.
- Rejected complexity: no SearXNG MCP fallback, no provider abstraction layer
  beyond the two direct modules, no new service process, no browser automation,
  no cookie/session persistence, no proxy rotation, no CAPTCHA handling, no
  private-network denylist, no persistent cache, and no prompt redesign.
- Evidence threshold: add any rejected mechanism only after a separate
  approved plan cites a concrete observed failure in direct search/read that
  focused tests and bounded direct HTTP cannot address.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside `config.py`,
  `rag/web_agent3`, named tests, and named docs as high-scrutiny changes.
- The responsible agent may remove SearXNG MCP references from active docs and
  tests because that removal is explicitly in scope.
- The responsible agent must search for existing equivalent helpers before
  creating extraction or config parsing code.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the
  plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent records current SearXNG MCP baseline before production-code edits.
   - Artifact:
     `development_plans/active/short_term/artifacts/searxng_mcp_baseline_2026-06-01.json`.
   - Baseline must contain the 10 cases named in `Contracts And Data Shapes`.
   - Record success count, latency summary, first non-empty line, output
     length, output SHA-256, and bounded excerpt for each case.
2. Parent adds focused failing tests in `tests/test_config.py` and
   `tests/test_web_agent3.py`.
   - Add `test_config_allows_empty_searxng_url`.
   - Add `test_config_reads_direct_web_settings_from_environment`.
   - Add `test_config_rejects_invalid_searxng_url`.
   - Add `test_web_agent3_search_reports_unavailable_without_searxng_url`.
   - Add `test_web_agent3_search_calls_direct_searxng_json_api`.
   - Add `test_web_agent3_search_formats_bounded_results`.
   - Add `test_web_agent3_url_read_sends_configured_browser_headers`.
   - Add `test_web_agent3_url_read_accepts_local_http_url`.
   - Add `test_web_agent3_url_read_rejects_non_http_scheme`.
   - Add `test_web_agent3_url_read_extracts_headings_sections_paragraphs`.
   - Add `test_web_agent3_url_read_caps_zero_max_length`.
   - Run the focused tests and record expected failures.
3. Parent starts one production-code subagent with this plan, mandatory skills,
   focused tests, and production boundary limited to `config.py` and
   `src/kazusa_ai_chatbot/rag/web_agent3/**`.
4. Production-code subagent implements config constants, `direct_searxng.py`,
   `url_reader.py`, and rewires `searxng_tools.py`.
5. Parent updates integration-style tests in `tests/test_web_agent3_routing.py`,
   `tests/test_e2e_live_llm.py`, and `tests/test_mcp_client.py`.
6. Parent reruns focused tests, then web-agent routing tests.
7. Parent updates README/HOWTO/web-agent ICD docs and static grep
   expectations.
8. Parent records direct-workflow 10-case regression comparison artifact.
9. Parent runs the full verification gate.
10. Parent starts one independent code-review subagent after verification
    passes.
11. Parent fixes review findings only inside approved change surface, reruns
   affected checks, and records evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.
- This plan is executed on branch `searxng-mcp-phaseout`; final completion
  does not merge to `main`.

## Progress Checklist

- [x] Stage 0 - SearXNG MCP baseline recorded.
  - Covers: implementation order step 1.
  - Verify: baseline artifact contains 10 cases, 10 successes, latency summary,
    and bounded output metadata.
  - Evidence: record artifact path and summary in `Execution Evidence`.
  - Sign-off: parent/2026-06-01 after evidence is recorded.
- [x] Stage 1 - Focused test contract established.
  - Covers: implementation order step 2.
  - Verify: named focused config and direct web tests fail for missing direct
    implementation or current MCP behavior.
  - Evidence: record commands and expected failures in `Execution Evidence`.
  - Sign-off: parent/2026-06-01 after evidence is recorded.
- [x] Stage 2 - Direct web facility implemented.
  - Covers: implementation order steps 3-4.
  - Verify: focused tests from Stage 1 pass.
  - Evidence: record changed production files and focused test output.
  - Sign-off: parent/2026-06-01 after evidence is recorded.
- [x] Stage 3 - Integration tests and docs updated.
  - Covers: implementation order steps 5-7.
  - Verify: web-agent routing tests, MCP client tests, static greps, and docs
    checks pass.
  - Evidence: record changed tests/docs and command output.
  - Sign-off: parent/2026-06-01 after evidence is recorded.
- [x] Stage 4 - Full verification complete.
  - Covers: implementation order steps 8-9.
  - Verify: direct-workflow regression artifact passes the baseline comparison,
    all commands in `Verification` pass, or a live LLM environment skip is
    explicitly accepted.
  - Evidence: record direct artifact path, comparison result, command output,
    and residual risks.
  - Sign-off: parent/2026-06-01 after evidence is recorded.
- [x] Stage 5 - Independent code review complete.
  - Covers: implementation order steps 10-11.
  - Verify: review subagent approves completion and affected checks rerun after
    fixes.
  - Evidence: record findings, fixes, reruns, and approval status.
  - Sign-off: parent/2026-06-01 after evidence is recorded.

## Verification

### Py Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\rag\web_agent3\direct_searxng.py src\kazusa_ai_chatbot\rag\web_agent3\url_reader.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py`
- `venv\Scripts\python -m py_compile tests\test_config.py tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_mcp_client.py tests\test_e2e_live_llm.py`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_config.py::TestDirectWebConfig::test_config_allows_empty_searxng_url tests\test_config.py::TestDirectWebConfig::test_config_reads_direct_web_settings_from_environment tests\test_config.py::TestDirectWebConfig::test_config_rejects_invalid_searxng_url -q`
- `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_search_reports_unavailable_without_searxng_url tests\test_web_agent3.py::test_web_agent3_search_calls_direct_searxng_json_api tests\test_web_agent3.py::test_web_agent3_search_formats_bounded_results tests\test_web_agent3.py::test_web_agent3_url_read_sends_configured_browser_headers tests\test_web_agent3.py::test_web_agent3_url_read_accepts_local_http_url tests\test_web_agent3.py::test_web_agent3_url_read_rejects_non_http_scheme tests\test_web_agent3.py::test_web_agent3_url_read_extracts_headings_sections_paragraphs tests\test_web_agent3.py::test_web_agent3_url_read_caps_zero_max_length -q`
- `venv\Scripts\python -m pytest tests\test_config.py tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_mcp_client.py -q`

### RAG/Web Regression Tests

- `venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_rag_initializer_cache2.py::test_rag_dispatcher_uses_deterministic_new_prefix tests\test_rag_initializer_cache2.py::test_rag_dispatcher_remaps_legacy_prefix_alias tests\test_rag_phase3_capability_agents.py tests\test_rag_phase3_supervisor_integration.py tests\test_quote_aware_rag_sequence.py -q`

### Static Greps

- `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read|mcp-searxng" src tests docs README.md`
  - Expected: no matches.
- `rg "MCP_SERVERS|MCP_CALL_TIMEOUT|MCP_CONNECT_TIMEOUT|mcp_manager" src tests docs README.md`
  - Expected: matches remain only for generic MCP config, generic MCP manager,
    service lifecycle, and generic MCP tests/docs. Matches must not describe
    SearXNG web search/read.
- `rg "192\\.168\\.2\\.10" src tests`
  - Expected: no matches. Runtime code and tests must not hard-code the
    operator's SearXNG host. Baseline artifacts may contain this address
    because they record measured external-state evidence.
- `rg "localhost:8080" src tests`
  - Expected: matches are allowed only for non-SearXNG debug UI documentation,
    such as `src/adapters/debug_adapter.py`. Any SearXNG config, search,
    URL-reader, or test fixture match is forbidden.

### Baseline Regression Comparison

- Record direct workflow performance using the 10 baseline cases and
  `SEARXNG_URL=http://192.168.2.10:8080` in the command environment only.
  Runtime code and tests must still read the value from `config.py`.
- Write
  `development_plans/active/short_term/artifacts/searxng_direct_regression_2026-06-01.json`.
- Compare the direct artifact against
  `development_plans/active/short_term/artifacts/searxng_mcp_baseline_2026-06-01.json`.
- Expected: all direct cases pass the direct workflow regression criteria in
  `Contracts And Data Shapes`.
- If any comparison fails, fix the regression and rerun this gate before
  sign-off.

### Live LLM/Web Smoke

- `venv\Scripts\python -m pytest tests\test_e2e_live_llm.py::test_live_web_agent3_returns_live_result -q -s -m "live_llm and live_db"`
  - Run only when live LLM and MongoDB configuration are available.
  - The test may skip when `SEARXNG_URL` is absent because the case is
    search-backed.
  - If it runs, inspect output one case at a time.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, URL-reader
  overreach, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  static checks, execution evidence, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `web_agent3` generic search calls direct SearXNG JSON API and never calls
  `mcp_manager` or `mcp-searxng` tools.
- `web_agent3` generic URL read is process-local, available by default, and
  does not require SearXNG or MCP.
- `SEARXNG_URL` absence does not crash import, service startup, package import,
  or web-agent graph execution.
- Search unavailable state is surfaced as bounded evidence/tool observation,
  not as an unhandled exception.
- URL reads allow local/private/intranet HTTP(S) resources and use configured
  browser-like request headers.
- Generic MCP support remains intact for non-SearXNG tools.
- Active source, tests, README, and HOWTO contain no SearXNG MCP setup or tool
  names.
- Direct-workflow 10-case comparison passes the baseline regression criteria,
  or the user explicitly accepts a recorded regression.
- All verification gates pass or record an accepted live environment skip.
- Independent code review is complete with no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Direct URL reader returns lower-quality text than MCP converter | Keep deterministic extraction bounded and test headings, paragraphs, section slicing, and char windows | Focused URL-reader tests and web-agent finalizer tests |
| SearXNG absence causes a graph-level failure | Return bounded unavailable observation from search tool | Focused absent-config test and WebAgent3 run test |
| Local URL access creates unbounded reads | Apply config-owned timeout, redirect, byte, and char caps | URL-reader limit tests |
| SearXNG MCP references remain hidden in active paths | Static greps over active source/tests/docs | Static grep gate |
| Generic MCP support regresses | Keep generic MCP client and tests with non-SearXNG examples | `tests/test_mcp_client.py` |

## Execution Evidence

- Draft created on 2026-06-01 with no production-code changes.
- Branch created: `searxng-mcp-phaseout`.
- Independent plan review completed by subagent `019e80f8-b00c-74b1-959f-834871cb775d`.
  Initial result: not approved for execution. Blockers were draft lifecycle
  status, missing 10-case baseline/regression gate, underspecified config
  defaults/bounds, overbroad SearXNG endpoint grep, missing no-merge rule, and
  ambiguous URL-reader argument support.
- Plan review blockers addressed before production-code edits: status and
  registry set to `approved`; baseline/regression artifact contract added;
  exact config defaults/bounds added; static grep expectations narrowed;
  branch/no-main rule added; URL-reader argument behavior made explicit.
- SearXNG direct service probe: `http://192.168.2.10:8080/search?...format=json`
  returned HTTP 200 JSON.
- SearXNG MCP protocol probe: `http://192.168.2.10:4001/mcp` returned HTTP
  400 to plain GET, which confirms endpoint reachability but not a valid MCP
  session by itself.
- SearXNG MCP baseline artifact:
  `development_plans/active/short_term/artifacts/searxng_mcp_baseline_2026-06-01.json`.
  Summary: 10 examples, 10 successes, 0 failures, MCP connect `193.64ms`,
  latency min `19.25ms`, median `616.93ms`, max `1413.68ms`, discovered tools
  `mcp-searxng__searxng_web_search` and `mcp-searxng__web_url_read`.
- Human-readable baseline summary:
  `development_plans/active/short_term/artifacts/searxng_mcp_baseline_2026-06-01.md`.
- Independent plan re-review completed by subagent `019e8101-97f4-7ed0-a8d8-4d704aef35e9`.
  Result: approved for execution; no blockers or important findings. Minor
  finding about direct artifact `mcp_url` ambiguity addressed by specifying
  `mcp_url=""` for direct artifacts.
- Stage 1 red config command:
  `$env:PYTHON_DOTENV_DISABLED='1'; venv\Scripts\python -m pytest tests\test_config.py::TestDirectWebConfig::test_config_allows_empty_searxng_url tests\test_config.py::TestDirectWebConfig::test_config_reads_direct_web_settings_from_environment tests\test_config.py::TestDirectWebConfig::test_config_rejects_invalid_searxng_url -q`.
  Result: 3 failed as expected. Failures prove `SEARXNG_URL` and related
  direct-web config constants are absent and invalid `SEARXNG_URL` is not
  rejected yet.
- Stage 1 red web command:
  `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_search_reports_unavailable_without_searxng_url tests\test_web_agent3.py::test_web_agent3_search_calls_direct_searxng_json_api tests\test_web_agent3.py::test_web_agent3_search_formats_bounded_results tests\test_web_agent3.py::test_web_agent3_url_read_sends_configured_browser_headers tests\test_web_agent3.py::test_web_agent3_url_read_accepts_local_http_url tests\test_web_agent3.py::test_web_agent3_url_read_rejects_non_http_scheme tests\test_web_agent3.py::test_web_agent3_url_read_extracts_headings_sections_paragraphs tests\test_web_agent3.py::test_web_agent3_url_read_caps_zero_max_length -q`
  with `PYTHON_DOTENV_DISABLED=1` and required route env vars set to
  placeholders. Result: 8 failed as expected. Failures prove
  `direct_searxng.py` and `url_reader.py` are absent and the old
  `mcp-searxng__web_url_read` path still handles reads.
- Stage 2 production-code worker subagent `019e8109-d082-7d52-a09e-5363c1eee09b`
  completed with status `DONE_WITH_CONCERNS`. Changed production files:
  `src/kazusa_ai_chatbot/config.py`,
  `src/kazusa_ai_chatbot/rag/web_agent3/direct_searxng.py`,
  `src/kazusa_ai_chatbot/rag/web_agent3/url_reader.py`,
  `src/kazusa_ai_chatbot/rag/web_agent3/searxng_tools.py`, and
  `src/kazusa_ai_chatbot/rag/web_agent3/subagent/generic.py`.
  Concern was limited to later-stage verification not being run by the worker.
- Stage 2 parent py_compile command:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\rag\web_agent3\direct_searxng.py src\kazusa_ai_chatbot\rag\web_agent3\url_reader.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py`.
  Result: passed.
- Stage 2 parent focused config command:
  `$env:PYTHON_DOTENV_DISABLED='1'; venv\Scripts\python -m pytest tests\test_config.py::TestDirectWebConfig::test_config_allows_empty_searxng_url tests\test_config.py::TestDirectWebConfig::test_config_reads_direct_web_settings_from_environment tests\test_config.py::TestDirectWebConfig::test_config_rejects_invalid_searxng_url -q`.
  Result: 3 passed.
- Stage 2 parent focused web command:
  `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_search_reports_unavailable_without_searxng_url tests\test_web_agent3.py::test_web_agent3_search_calls_direct_searxng_json_api tests\test_web_agent3.py::test_web_agent3_search_formats_bounded_results tests\test_web_agent3.py::test_web_agent3_url_read_sends_configured_browser_headers tests\test_web_agent3.py::test_web_agent3_url_read_accepts_local_http_url tests\test_web_agent3.py::test_web_agent3_url_read_rejects_non_http_scheme tests\test_web_agent3.py::test_web_agent3_url_read_extracts_headings_sections_paragraphs tests\test_web_agent3.py::test_web_agent3_url_read_caps_zero_max_length -q`
  with `PYTHON_DOTENV_DISABLED=1` and required route env vars set to
  placeholders. Result: 8 passed.
- Stage 3 integration command:
  `venv\Scripts\python -m pytest tests\test_web_agent3_routing.py tests\test_mcp_client.py -q`
  with `PYTHON_DOTENV_DISABLED=1` and required route env vars set to
  placeholders. Result: 13 passed.
- Stage 3 static grep:
  `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read|mcp-searxng" src tests docs README.md`.
  Result: no matches.
- Stage 3 static grep:
  `rg "MCP_SERVERS|MCP_CALL_TIMEOUT|MCP_CONNECT_TIMEOUT|mcp_manager" src tests docs README.md`.
  Result: matches remain only for generic MCP config, MCP manager lifecycle,
  generic MCP tests, and generic MCP docs; no match describes SearXNG web
  search/read.
- Stage 3 static grep:
  `rg "192\.168\.2\.10" src tests`. Result: no matches.
- Stage 3 static grep:
  `rg "localhost:8080" src tests`. Result: one allowed debug UI reference in
  `src/adapters/debug_adapter.py`.
- Stage 4 initial direct regression comparison exposed two URL-reader
  extraction regressions (`read_02`, `read_04`) caused by HTML parser handling
  of metadata and link/button-only text. Fixed by treating `meta`/`link` as
  void metadata and extracting `a`/`button` text where it is visible content.
  Rerun result before later search-parameter fix: 10 direct cases, 10
  successes, median `567.665ms`.
- Stage 4 live WebAgent3 smoke exposed a direct-search regression: the direct
  SearXNG client sent blank optional `time_range` and `language` parameters,
  and the observed SearXNG instance returned HTTP 400. Fixed by omitting
  `time_range` and `language` unless non-empty after stripping whitespace.
  Added `test_web_agent3_search_omits_empty_optional_searxng_params`.
- Final Stage 4 direct search probe with
  `SEARXNG_URL=http://192.168.2.10:8080`: direct `web_search` returned normal
  results for `Auckland weather today`, `今天 奥克兰 天气`, and `奥克兰天气`.
- Final Stage 4 py_compile command:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\rag\web_agent3\direct_searxng.py src\kazusa_ai_chatbot\rag\web_agent3\url_reader.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py tests\test_config.py tests\test_web_agent3.py tests\test_mcp_client.py tests\test_e2e_live_llm.py`.
  Result: passed.
- Final Stage 4 focused batch:
  `venv\Scripts\python -m pytest tests\test_config.py tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_mcp_client.py -q`
  with `PYTHON_DOTENV_DISABLED=1` and required route env vars set to the local
  OpenAI-compatible endpoint. Result after review fixes: 88 passed.
- Final Stage 4 RAG/web regression command:
  `venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_rag_initializer_cache2.py::test_rag_dispatcher_uses_deterministic_new_prefix tests\test_rag_initializer_cache2.py::test_rag_dispatcher_remaps_legacy_prefix_alias tests\test_rag_phase3_capability_agents.py tests\test_rag_phase3_supervisor_integration.py tests\test_quote_aware_rag_sequence.py -q`
  with `PYTHON_DOTENV_DISABLED=1`, `SEARXNG_URL=http://192.168.2.10:8080`,
  and required route env vars set to the local OpenAI-compatible endpoint.
  Result: 133 passed.
- Final Stage 4 static greps rerun:
  `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read|mcp-searxng" src tests docs README.md`
  returned no matches; generic MCP grep retained only generic MCP config,
  lifecycle, tests, and docs; `rg "192\.168\.2\.10" src tests` returned no
  matches; `rg "localhost:8080" src tests` returned only
  `src/adapters/debug_adapter.py`.
- Final Stage 4 direct regression artifact:
  `development_plans/active/short_term/artifacts/searxng_direct_regression_2026-06-01.json`.
  Summary after review fixes: 10 examples, 10 successes, 0 failures, latency
  min `147.07ms`, median `607.205ms`, max `1252.97ms`, baseline median
  `616.93ms`, median
  limit `771.16ms`, criteria error count `0`.
- Stage 5 independent code review subagent
  `019e8123-9196-7700-9b2a-7e777c294ff1` found three issues: malformed
  HTTP(S) URLs could raise through `web_url_read`, URL-reader byte caps were
  enforced after full response download, and the live WebAgent3 smoke had a
  DB fixture dependency that blocked the planned live command.
- Review fixes: `web_url_read` now catches malformed URL parsing and
  `httpx.InvalidURL` as bounded `Error:` observations; URL reads now use
  `httpx.AsyncClient.stream(...)` and stop reading once
  `WEB_URL_READ_MAX_BYTES` would be exceeded; deterministic tests were added
  for malformed URLs, `httpx.InvalidURL`, and streaming byte-cap enforcement;
  `test_live_web_agent3_returns_live_result` now checks the LLM endpoint
  directly and no longer depends on the DB-backed `live_env` fixture.
- Review-fix targeted command:
  `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_url_read_returns_error_for_malformed_http_url tests\test_web_agent3.py::test_web_agent3_url_read_returns_error_for_httpx_invalid_url tests\test_web_agent3.py::test_web_agent3_url_read_stops_stream_when_response_exceeds_cap -q`.
  Result: 3 passed.
- Review-fix full web-agent command:
  `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`.
  Result: 32 passed.
- Live pytest smoke command:
  `venv\Scripts\python -m pytest tests\test_e2e_live_llm.py::test_live_web_agent3_returns_live_result -m "live_llm and live_db" -q -s`
  with local OpenAI-compatible route env vars,
  `SEARXNG_URL=http://192.168.2.10:8080`, and `MCP_SERVERS={}`. Result after
  removing the unnecessary DB fixture dependency from this web-only case:
  1 passed.
- Stage 5 re-review by subagent `019e8123-9196-7700-9b2a-7e777c294ff1`:
  no blockers or important findings remain. The re-review confirmed malformed
  URL handling, `httpx.InvalidURL` handling, streaming byte-cap enforcement,
  deterministic test coverage, removal of the DB fixture dependency from the
  live WebAgent3 smoke, no hidden active SearXNG MCP usage, and preserved
  generic MCP capability. Residual non-blocking risk: this live smoke remains
  in a module globally marked `live_db`, so the command still filters
  `live_llm and live_db` even though the case itself is web/LLM-only.
