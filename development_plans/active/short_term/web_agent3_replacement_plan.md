# web_agent3 Replacement Plan

Status: in_progress
Class: large
Created: 2026-05-25
Last updated: 2026-05-27
Owner: Codex

## Summary

Build `web_agent3` as the replacement candidate for `web_search_agent2`, keep
`web_search_agent2` active until `web_agent3` is implemented and verified, then
perform one later big-bang transition.

Latest user correction:

- Ordinary webpage search and URL reads must go through the existing SearXNG
  facility.
- Do not over design.
- Runtime LLM prompts must follow project prompt rules and preserve stable
  prompt prefixes for local LLM cache behavior.
- The web-agent router is the first LLM stage. It emits only `action`,
  `source`, and `query`; source-specific parsing belongs inside subagents.
- YouTube and Bilibili remain source subagent placeholders. nHentai is approved
  for Stage 3C real metadata/search implementation inside its source subagent.
- Source subagents are tracked as separate modules under `web_agent3/subagent/`:
  `generic.py`, `bilibili.py`, `youtube.py`, and `nhentai.py`. Each module
  exposes its source name, prompt-facing description, and primary async
  execution interface. The web agent auto-discovers these subagent modules.
- Subagent `DESCRIPTION` values carry source-local query generation rules.
  The router prompt remains generic and tells the model to follow the selected
  source description when producing `query`.
- LLM-facing source descriptions expose only source capability and query
  shaping rules. Transport details, placeholder implementation notes, retry
  control fields, and comparison-fixture metadata stay out of LLM payloads.
- nHentai Stage 3C scope is metadata-only: `read` returns gallery name/title
  and tags; `search` returns bounded gallery candidates. No download, image
  page read, thumbnail payload, comment read, favorite mutation, account
  mutation, moderation endpoint, or media binary handling is in scope.
- The side-by-side real LLM comparison must be apple-to-apple through the
  shared public helper contract: both agents receive the same `task`,
  `context`, and `max_attempts`, and both are judged from the same public
  output shape.

## Current Contract

`web_agent3` must keep the same helper input/output contract as
`web_search_agent2`:

```python
async def run(
    self,
    task: str,
    context: dict[str, Any],
    max_attempts: int = 3,
) -> dict[str, Any]:
```

Output:

```python
{
    "resolved": bool,
    "result": str,
    "attempts": 1,
    "cache": {
        "enabled": False,
        "hit": False,
        "cache_name": "",
        "reason": "agent_not_cacheable",
    },
}
```

`result` stays a source-grounded evidence string for RAG projection. The RAG
supervisor, cognition, dialog, persistence, adapters, scheduler, and database
contracts do not change.

## Mandatory Rules

- Do not read `.env`.
- Use `venv\Scripts\python` for Python and pytest commands.
- Use `apply_patch` for manual edits.
- Keep `web_search_agent2` active until all `web_agent3` functions and the
  comparison suite have been verified.
- Do not add a service process, dependency, MCP server, global config field,
  custom URL fetcher, URL safety layer, or HTML extraction layer.
- The only approved direct provider HTTP path in pre-cutover `web_agent3` is
  the nHentai Stage 3C API client code inside
  `src/kazusa_ai_chatbot/rag/web_agent3/subagent/nhentai.py`. Generic web
  search and generic URL reads still must use the existing SearXNG facility.
- `NHENTAI_TOKEN` may be read from the process environment inside
  `subagent/nhentai.py` only. Do not read `.env`, do not log the token, do not
  put the token in prompts or observations, and do not add a required
  `config.py` import-time variable that can break unrelated startup.
- All generic web search and generic URL-read execution in `web_agent3` must
  call the existing SearXNG MCP tools through the existing local facility:
  `mcp-searxng__searxng_web_search` and `mcp-searxng__web_url_read`.
- Bilibili and YouTube adapters are allowed only as router placeholders. They
  must return explicit no-search-data observations without
  falling back to generic SearXNG, and carry:
  `FIXME(web_agent3): replace no-search-data placeholder with provider API client in a future approved plan.`
- nHentai must stop being a placeholder in Stage 3C and must use only the
  official nHentai API v2 metadata/search endpoints needed by this plan.
- Runtime prompts are Chinese-first RAG evidence prompts. Stable role,
  generation procedure, and output contract text stay in `SystemMessage`;
  runtime-varying fields such as `task`, projected `context`,
  `reference_time`, tool history, and evaluator feedback stay in
  `HumanMessage` JSON.
- Router/generator prompts must not include a `# 输入格式` or `# Input Format`
  section. Field meanings must be explained through concise generation rules
  and the output contract.
- Router/generator source descriptions must not mention SearXNG, placeholder
  adapter execution details, fallback behavior, credentials, transport, or
  provider metadata. Those are executor/subagent/code concerns.
- The router/generator output contract has exactly three fields:
  `action`, `source`, and `query`. The `source` value must be one available
  source token listed in the source-adapter descriptions rendered in the
  generator prompt. Do not add URL fields, target-type fields, reasons,
  provider metadata, or API-specific parameters.
- After context compaction or major checklist sign-off, reread this plan before
  continuing.

## Router Decision

`web_agent3` uses one LLM router/generator stage before execution. It receives
the current `Web-evidence:` task, compact runtime context, reference time,
bounded prior tool history, and evaluator feedback. It returns only:

```json
{
    "action": "search",
    "source": "generic",
    "query": "local tool router demo web agent architecture"
}
```

Allowed values:

- `action`: `search`, `read`, or `stop`.
- `source`: `generic`, `bilibili`, `youtube`, or `nhentai`.
- `query`: the only payload passed to the selected subagent.

For `search`, `query` is a ready-to-run search string for the selected source.
For `read`, `query` carries the raw target string such as a URL, BV/AV value,
YouTube URL/video text, nHentai numeric id, or other user-provided target.
For `stop`, `query` is an empty string.

The router/generator must not extract YouTube video IDs, Bilibili BV/AV IDs,
nHentai gallery IDs, API fields, credentials, request parameters, or transport
details. Those are source-adapter responsibilities. Bilibili and YouTube
placeholder adapters return no search data. The nHentai Stage 3C adapter may
extract gallery ids deterministically from its `query` with regex and may call
the official API v2 endpoints named in this plan.

## Source Subagent Decision

- `generic` is the only active web search and URL-read executor in the current
  stage. It uses the existing SearXNG facility.
- `youtube`, `bilibili`, and `nhentai` must be present in the source adapter
  registry as dedicated subagents.
- Source subagents are not tracked in a single `source_subagents.py` file.
  They live under `subagent/` with one source module per file.
- Bilibili and YouTube remain placeholder subagents. Each placeholder receives
  the router decision with `query` unchanged and returns an explicit no-result
  observation.
- Placeholder subagents must not call SearXNG, must not delegate to `generic`,
  and must not perform automatic source fallback.
- nHentai receives the router decision with `query` unchanged, then owns
  deterministic extraction of gallery ids, API parameter shaping, HTTP request
  execution, and response compaction inside `subagent/nhentai.py`.
- nHentai `read` must read only gallery name/title and grouped tags. It must
  not expose image page URLs, thumbnails, download URLs, comments, favorite
  state, account state, moderation data, or media binary data.
- nHentai `search` must search galleries through the official search endpoint
  and return bounded gallery candidates suitable for evaluator follow-up. It
  must not download or read pictures.
- The evaluator may inspect the no-result observation and decide whether the
  overall web-agent loop should stop or continue, but source adapters must not
  silently substitute another source.
- Real Bilibili and YouTube provider APIs, credentials, request parameter
  extraction, and local or MCP tool variants remain deferred to a later
  approved plan.

## Target Architecture

```text
RAG supervisor
  -> WebAgent3.run(task, context, max_attempts=3)
       -> LangGraph router/evaluator/finalizer loop
       -> router/generator emits action/source/query
       -> executor selects subagent from source
          -> generic_adapter  -> existing SearXNG MCP facility
          -> bilibili_adapter -> no_search_data placeholder
          -> youtube_adapter  -> no_search_data placeholder
          -> nhentai_adapter  -> official nHentai API v2 metadata/search
       -> same text evidence result contract
```

The router is inside the web helper. The outer RAG supervisor still sees one
web helper and one text evidence result. The executor dispatches only by
`source` and `action`; it must not reinterpret the semantic meaning of `query`.

## Module Layout

Final candidate code lives under:

```text
src/kazusa_ai_chatbot/rag/web_agent3/
  README.md
  __init__.py
  agent.py
  contracts.py
  providers.py
  searxng_tools.py
  subagent/
    __init__.py
    generic.py
    bilibili.py
    youtube.py
    nhentai.py
```

Ownership:

| File | Responsibility |
|---|---|
| `README.md` | Subpackage ICD. |
| `agent.py` | Router/evaluator/finalizer loop and `WebAgent3`. |
| `searxng_tools.py` | Existing SearXNG MCP tool calls. |
| `providers.py` | Thin source decision executor that dispatches to registered source subagents. |
| `subagent/__init__.py` | Auto-discovery and validation for source subagent modules. |
| `subagent/generic.py` | Generic search/read subagent backed by the existing SearXNG facility. |
| `subagent/bilibili.py` | Bilibili placeholder subagent. |
| `subagent/youtube.py` | YouTube placeholder subagent. |
| `subagent/nhentai.py` | nHentai API v2 metadata/search subagent. |
| `contracts.py` | Minimal router output and comparison-test data contracts. |

## Contracts And Data Shapes

### nHentai Subagent Input

`subagent/nhentai.py` keeps the existing source subagent interface:

```python
async def execute(decision: _RouterDecision) -> dict[str, Any]:
```

`decision.query` remains the only source-specific payload. The router does not
produce API fields.

### nHentai Read Behavior

Accepted `read` targets:

- a bare gallery id such as `652244`;
- a gallery page URL such as `https://nhentai.net/g/652244/`;
- an API URL/path containing `/api/v2/galleries/652244`.

Implementation instruction:

- Extract the gallery id with a deterministic regex inside
  `subagent/nhentai.py`.
- Call `GET https://nhentai.net/api/v2/galleries/{gallery_id}`.
- If `NHENTAI_TOKEN` is present, send `Authorization: Key <token>`.
- Always send a descriptive `User-Agent`.
- Return only a compact observation with:

```python
{
    "status": "success" | "not_found" | "error",
    "source": "nhentai",
    "action": "read",
    "query": str,
    "gallery_id": int | None,
    "url": str | None,
    "title": {
        "english": str,
        "japanese": str | None,
        "pretty": str,
    },
    "tags": {
        "language": list[str],
        "artist": list[str],
        "group": list[str],
        "parody": list[str],
        "character": list[str],
        "tag": list[str],
        "category": list[str],
    },
    "message": str,
}
```

Forbidden read fields: page image URLs, cover paths, thumbnail URLs, download
URLs, comments, favorite state, user/account fields, moderation fields, raw
headers, and tokens.

### nHentai Search Behavior

Accepted `search` targets:

- normal search text;
- nHentai search syntax supported by the official API, including tag filters,
  exact phrases, negation, numeric filters, and date filters;
- a numeric gallery id, which the nHentai subagent may internally treat as a
  gallery `read` because the router is not responsible for that distinction.

Implementation instruction:

- Call `GET https://nhentai.net/api/v2/search` with `query=<decision.query>`,
  `sort=date`, and `page=1`.
- Return at most five gallery candidates.
- Each candidate contains only:

```python
{
    "id": int,
    "url": str,
    "title": str,
    "num_pages": int,
    "num_favorites": int,
}
```

Search observations must not include image URLs, thumbnail URLs, download
URLs, or token/header data.

## Change Surface

Current pre-cutover work may touch only:

- `development_plans/active/short_term/web_agent3_replacement_plan.md`
- `src/kazusa_ai_chatbot/rag/web_agent3/`
- `tests/test_web_agent3.py`
- `tests/test_web_agent3_routing.py`
- `tests/test_web_agent3_nhentai.py`
- `tests/test_web_agent_comparison_live_llm.py`
- ignored debug-LLM artifacts under `test_artifacts/`

Later big-bang transition may touch only active web-helper references in:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/live_context_agent.py`
- active web/RAG tests and docs that name `web_search_agent2`
- deletion of `src/kazusa_ai_chatbot/rag/web_search_agent.py`

No other modules are in scope.

## Implementation Order

1. Keep `web_search_agent2` active.
2. Implement `web_agent3` functions inside the package with the
   `action/source/query` router contract.
3. Run focused deterministic `web_agent3` tests.
3A. Update code to match the final router/subagent architecture:
   - In `agent.py`, replace the current bound-tool generator with a structured
     router/generator LLM stage that parses exactly `action`, `source`, and
     `query`.
   - In `agent.py`, keep the evaluator and finalizer loop; evaluator feedback
     returns to the router/generator on continuation.
   - In `agent.py`, remove provider-route metadata from prompt-facing state and
     final helper output. The public result remains only `resolved`, `result`,
     `attempts`, and `cache`.
   - In `providers.py`, implement source adapter dispatch where
     `generic`, `bilibili`, `youtube`, and `nhentai` each receive `query`
     unchanged. The non-generic placeholder adapters return no search data
     with the fixed FIXME marker and do not call SearXNG.
   - In `searxng_tools.py`, keep existing SearXNG MCP calls as the only web
     search and URL-read execution path.
   - In `contracts.py`, keep only minimal router decision and test fixture
     contracts needed by `web_agent3`; do not introduce provider metadata,
     locators, URL fields, API parameter objects, or credential/config shapes.
   - In `README.md`, update the ICD to describe the
     `router -> executor -> subagent -> evaluator -> loop/finalizer` flow.
   - In `tests/test_web_agent3.py`, add focused tests proving strict router
     output parsing, prompt omission of `# 输入格式` / `# Input Format`, query
     pass-through to each source adapter, placeholder no-search-data behavior,
     evaluator-driven loop continuation, and public helper contract parity.
3B. Add dedicated placeholder source subagents:
   - Add `subagent/` so each source subagent is tracked in its own module:
     `generic.py`, `bilibili.py`, `youtube.py`, and `nhentai.py`.
   - Each subagent module exposes its source name, prompt-facing description,
     and primary async execution interface.
   - Source-specific query generation rules live in each subagent
     `DESCRIPTION`; the router prompt only tells the model to follow the
     selected source description.
   - In `subagent/__init__.py`, auto-discover and validate the subagent
     modules, then expose the source description map and execution registry.
   - In `providers.py`, keep only the selected source decision execution
     surface used by the graph executor.
   - Each placeholder subagent receives the router decision with `query`
     unchanged and returns an explicit no-result observation.
   - Placeholder subagents must not call SearXNG, must not delegate to
     `generic`, and must not perform automatic fallback.
   - Keep `generic` as the only active SearXNG-backed search/read source in
     this stage.
   - In `agent.py`, keep the executor dispatch limited to selected source
     adapter execution; do not add router-side compensation for placeholder
     no-result observations.
   - In `tests/test_web_agent3.py`, add or keep focused tests proving
     placeholder no-result behavior, no SearXNG call for non-generic sources,
     and unchanged query delivery into the placeholder subagents.
3C. Implement nHentai metadata/search subagent:
   - In `tests/test_web_agent3_nhentai.py`, add focused deterministic tests
     before production code for gallery id extraction from bare ids, gallery
     URLs, API URLs, and invalid targets.
   - In `tests/test_web_agent3_nhentai.py`, add tests that `read` calls the
     gallery endpoint and returns only name/title plus grouped tags.
   - In `tests/test_web_agent3_nhentai.py`, add tests that `search` calls the
     gallery search endpoint and returns bounded gallery candidates.
   - In `tests/test_web_agent3_nhentai.py`, add tests that a numeric `search`
     target is handled as a gallery lookup inside the nHentai subagent.
   - In `tests/test_web_agent3_nhentai.py`, add tests for missing token,
     present token header, 404, 429, timeout/HTTP error, malformed JSON or
     malformed response shape, and no leakage of token, headers, image URLs,
     thumbnail URLs, download URLs, comments, or favorite state.
   - In `subagent/nhentai.py`, replace the placeholder with deterministic
     helper functions for gallery id extraction, auth header creation,
     official API request execution using existing `httpx`, response
     compaction, and bounded error observations.
   - In `subagent/nhentai.py`, read `NHENTAI_TOKEN` with `os.getenv` at
     execution time. Do not add a required `config.py` setting.
   - In `subagent/nhentai.py`, keep prompt-facing `DESCRIPTION` source-local
     and query-oriented. It may mention gallery metadata/search and query
     shaping, but must not mention token, authorization headers, endpoints,
     implementation placeholders, or transport mechanics.
   - In `tests/test_web_agent3.py`, update previous placeholder expectations
     so only Bilibili and YouTube remain no-data placeholders.
   - In `tests/test_web_agent3_routing.py`, keep router and executor boundary
     tests focused on source dispatch and query pass-through; do not make the
     router extract nHentai ids.
4. Run the focused deterministic Stage 3A, Stage 3B, and Stage 3C tests and
   static checks.
5. Run the same ten real LLM comparison cases one at a time and inspect traces.
6. Write a debug-LLM comparison report from real trace data.
7. Complete RCA against `web_search_agent2` and close parity gaps.
8. Only after focused tests, live comparison, RCA, and review are acceptable,
   perform the big-bang transition in a separate pass.

## Verification

Focused deterministic:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\__init__.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\youtube.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py
venv\Scripts\python -m py_compile tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py tests\test_web_agent_comparison_live_llm.py
venv\Scripts\python -m pytest tests\test_web_agent3.py -q
venv\Scripts\python -m pytest tests\test_web_agent3_nhentai.py -q
venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py -q
```

Focused prompt/payload minimization must prove:

- router source descriptions expose capability and query guidance only, not
  transport or placeholder execution details;
- evaluator LLM payload does not receive retry/cap control fields;
- finalizer LLM payload receives clean evaluator feedback, not message-wrapper
  metadata.

Static checks:

```powershell
rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py
```

Expected: no matches.

```powershell
rg "httpx|NHENTAI_TOKEN|Authorization" src\kazusa_ai_chatbot\rag\web_agent3
```

Expected: matches only in `subagent/nhentai.py`.

```powershell
rg "NHENTAI_TOKEN|Authorization" tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py
```

Expected: matches only in `tests/test_web_agent3_nhentai.py`.

```powershell
rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3
```

Expected: matches only in `searxng_tools.py`.

```powershell
rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py
```

Expected: no matches, except `selected_provider_name` and
`executed_provider_name` may appear only inside comparison-fixture structures
if those fixtures still mirror old trace artifacts; runtime router/executor
code must not depend on them.

Real LLM comparison:

Run each test in `tests/test_web_agent_comparison_live_llm.py` individually
with `-q -s -m live_llm`, inspect the emitted trace, then proceed to the next
case. Do not batch-run the live LLM suite as the evidence source. The suite
must compare `WebSearchAgent().run(...)` and `WebAgent3().run(...)` with the
same public intake and output contract; internal finalizer-only comparison is
not sufficient for cutover evidence.

## Progress Checklist

- [x] Stage 1 - `web_search_agent2` restored as active runtime helper.
  - Evidence: prior focused checks passed with active old-agent wiring.
  - Sign-off: Codex / 2026-05-26.

- [x] Stage 2 - `web_agent3` package and ICD created.
  - Evidence: `src/kazusa_ai_chatbot/rag/web_agent3/README.md` exists.
  - Sign-off: Codex / 2026-05-26.

- [x] Stage 3 - direct HTTP implementation removed and SearXNG facility path
  restored inside `web_agent3`.
  - Evidence: focused deterministic tests pass after rewrite.
  - Sign-off: Codex / 2026-05-26.

- [x] Stage 3A - router/generator and subagent dispatch code updated.
  - Code changes: `src/kazusa_ai_chatbot/rag/web_agent3/agent.py`,
    `contracts.py`, `providers.py`, `searxng_tools.py`, `README.md`, and
    `tests/test_web_agent3.py`; comparison fixture constructor updates in
    `tests/test_web_agent_comparison_live_llm.py`.
  - Verify: focused deterministic Stage 3A tests pass; static checks show no
    direct HTTP stack and no router prompt `# 输入格式` / `# Input Format`;
    executor passes `query` unchanged to selected source adapters.
  - Evidence: focused deterministic checks and static checks passed.
  - Sign-off: Codex / 2026-05-26.

- [x] Stage 3B - dedicated placeholder source subagents tracked and verified.
  - Code changes: `src/kazusa_ai_chatbot/rag/web_agent3/subagent/`,
    `providers.py`, `agent.py` if executor dispatch requires adjustment,
    `contracts.py` if discovered source validation requires adjustment,
    `README.md` if the ICD needs clarification, and `tests/test_web_agent3.py`.
  - Verify: focused deterministic tests prove `youtube`, `bilibili`, and
    `nhentai` return no-result observations, do not call SearXNG, do not
    delegate to `generic`, receive `query` unchanged, and are tracked outside
    the thin executor facade as per-source modules discovered from
    `subagent/`.
  - Evidence: focused tests, full `tests/test_web_agent3.py`, adjacent
    `tests/test_web_search_agent.py tests/test_web_agent3.py`, and static
    greps passed.
  - Sign-off: Codex / 2026-05-27.

- [x] Stage 3C - nHentai metadata/search subagent implemented and verified.
  - Code changes: `src/kazusa_ai_chatbot/rag/web_agent3/subagent/nhentai.py`,
    `src/kazusa_ai_chatbot/rag/web_agent3/README.md` if ICD wording needs
    update, `tests/test_web_agent3.py`, `tests/test_web_agent3_routing.py`,
    and `tests/test_web_agent3_nhentai.py`.
  - Verify: focused tests prove read returns only title/name and grouped tags,
    search returns bounded gallery candidates, token headers are used only
    when `NHENTAI_TOKEN` is present, numeric search is handled inside the
    nHentai subagent, and no image/download/comment/favorite/token/header data
    leaks into observations.
  - Evidence: focused `tests/test_web_agent3_nhentai.py`, adjacent web-agent
    regression batch, static greps, and independent Stage 3C code review
    passed. Live API smoke was not run because the user did not request it.
  - Sign-off: Codex / 2026-05-27.

- [ ] Stage 4 - parity comparison rerun and RCA report updated.
  - Evidence: ten live LLM cases run one at a time, traces inspected, and a
    debug-LLM report written from real data.
  - Sign-off: `<agent/date>`.

- [ ] Stage 5 - big-bang transition completed.
  - Evidence: active wiring/docs/tests use `web_agent3`, old helper deleted,
    integration tests pass.
  - Sign-off: `<agent/date>`.

- [ ] Stage 6 - independent code review and final verification complete.
  - Evidence: review findings, fixes, reruns, and residual risks recorded.
  - Sign-off: `<agent/date>`.

## Execution Evidence

- 2026-05-26 restore-and-compare hold:
  - Active runtime wiring was restored to `web_search_agent2`.
  - Added `tests/test_web_agent_comparison_live_llm.py` with ten real LLM
    comparison cases.
  - Prior report:
    `test_artifacts/llm_reviews/web_agent2_vs_web_agent3_live_llm_review_20260526.md`.
  - RCA report:
    `test_artifacts/llm_reviews/web_agent3_transition_rca_20260526.md`.
- 2026-05-26 latest correction:
  - Replaced the over-designed direct HTTP/search/fetch package with a smaller
    SearXNG-facility implementation.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 12 passed.
- 2026-05-26 Stage 3A planning update:
  - Added concrete Stage 3A code-change scope for the strict
    `action/source/query` router contract, source subagent dispatch, prompt
    omission of `# 输入格式` / `# Input Format`, focused tests, and static
    checks.
- 2026-05-26 Stage 3A implementation:
  - Updated `web_agent3` to use
    `router/generator -> executor -> source subagent -> evaluator -> loop/finalizer`.
  - Router/generator parses only `action`, `source`, and `query`.
  - Source subagents receive `query` unchanged; non-generic sources initially
    fell back to the generic SearXNG path with the fixed FIXME marker.
  - Removed provider-route metadata from runtime state and focused tests.
  - Updated the subpackage ICD and comparison fixture constructors.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 14 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py -q`: 20 passed.
  - `venv\Scripts\python -m pytest -m live_llm tests\test_web_agent_comparison_live_llm.py --collect-only -q`: 10 tests collected.
  - Static check `rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|httpx|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`: matches only `searxng_tools.py`.
  - Static check `rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "selected_provider_name|executed_provider_name|provider_history" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent_comparison_live_llm.py`: no matches.
  - Live LLM comparison and big-bang cutover were not run in Stage 3A.
- 2026-05-27 Stage 3A placeholder-adapter prompt update:
  - Added explicit Bilibili, YouTube, and nHentai placeholder adapters in the
    source adapter registry.
  - Non-generic placeholder adapters now return `no_search_data` without
    calling generic SearXNG.
  - Updated the router/generator prompt to use the existing project prompt
    style: one static triple-quoted prompt string, Chinese-first instructions,
    `# 来源原则`, `# 审计步骤`, and `# 输出格式`.
  - Source principles are rendered from the available source adapter
    descriptions with `.format(...)`; the output shape keeps `source` as a
    string instead of duplicating the current source enum.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_router_prompt_uses_project_prompt_style tests\test_web_agent3.py::test_web_agent3_specialized_adapters_return_no_search_data -q`: 2 passed after failing for the expected missing behavior.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 15 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py -q`: 21 passed.
  - Static check `rg "_WEB_AGENT3_GENERATOR_PROMPT\s*=\s*\(|_WEB_AGENT3_GENERATOR_PROMPT\s*=\s*f|= f'''|= f\"\"\"|source adapter roster|# 输出契约|locator|provider metadata" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`: matches only `searxng_tools.py`.
- 2026-05-27 Stage 3B planning update:
  - Added a dedicated Stage 3B implementation/order checkpoint for
    `youtube`, `bilibili`, and `nhentai` placeholder subagents.
  - Stage 3B tracks the explicit no-result, no-SearXNG, no-generic-fallback
    behavior separately from Stage 3A router/generator work.
  - No production code was changed for this planning update.
- 2026-05-27 Stage 3B implementation:
  - Added focused coverage proving `bilibili`, `youtube`, and `nhentai` are
    dedicated placeholder source subagents separate from `generic`.
  - Verified placeholder source subagents return `no_search_data`, preserve
    the router `query`, do not call SearXNG, and do not delegate to `generic`.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_placeholder_sources_are_dedicated_no_result_subagents tests\test_web_agent3.py::test_web_agent3_specialized_adapters_return_no_search_data -q`: 2 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 16 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py -q`: 22 passed.
  - Static check `rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|httpx|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`: matches only `searxng_tools.py`.
  - Static check `rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name|provider metadata" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - CJK quote safety check `rg '[“”]' src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py`: no matches.
- 2026-05-27 Stage 3B source-module correction:
  - User corrected the subagent tracking shape: `source_subagents.py` was too
    shallow; the required layout is `subagent/generic.py`,
    `subagent/bilibili.py`, `subagent/youtube.py`, and
    `subagent/nhentai.py`.
  - Added a failing deterministic test for per-source module discovery before
    changing production code. The test failed on missing `subagent/`.
  - Replaced `source_subagents.py` with `subagent/` package modules.
  - `subagent/__init__.py` now auto-discovers and validates subagent modules,
    exposes the source description map for the router prompt, and exposes the
    execution registry for the provider facade.
  - Each subagent module exposes `SOURCE`, `DESCRIPTION`, and `execute(...)`.
  - `providers.py` remains the thin graph-facing dispatch facade and does not
    own source-specific registry data.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_source_subagents_are_discovered_from_subagent_package tests\test_web_agent3.py::test_web_agent3_generic_search_receives_query_unchanged tests\test_web_agent3.py::test_web_agent3_placeholder_sources_are_dedicated_no_result_subagents tests\test_web_agent3.py::test_web_agent3_specialized_adapters_return_no_search_data -q`: 4 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\__init__.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\youtube.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 17 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py -q`: 23 passed.
  - Static check `rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|httpx|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`: matches only `searxng_tools.py`.
  - Static check `rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name|provider metadata" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - CJK quote safety check `rg '[“”]' src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\subagent tests\test_web_agent3.py`: no matches.
  - File presence check `Test-Path -LiteralPath 'src/kazusa_ai_chatbot/rag/web_agent3/source_subagents.py'`: `False`.
- 2026-05-27 Stage 3B source-description query guidance:
  - Moved generic web search/read query generation guidance into
    `subagent/generic.py` `DESCRIPTION`.
  - Added placeholder-source query guidance to `subagent/bilibili.py`,
    `subagent/youtube.py`, and `subagent/nhentai.py` descriptions.
  - Updated the router prompt to require `query` to follow the selected source
    description while keeping router output limited to `action`, `source`, and
    `query`.
  - Sorted subagent discovery by module name for stable source-description
    rendering.
  - Added focused deterministic coverage for subagent-owned generation rules.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_router_uses_subagent_generation_rules -q`: failed before implementation for missing generic rules, then passed after implementation.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\__init__.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\youtube.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 18 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py -q`: 24 passed.
  - Static check `rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|httpx|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`: matches only `searxng_tools.py`.
  - Static check `rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name|provider metadata" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - CJK quote safety check `rg '[“”]' src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\subagent tests\test_web_agent3.py`: no matches.
- 2026-05-27 LLM-input minimization pass:
  - Removed SearXNG transport detail from the router-visible generic source
    description.
  - Removed placeholder/no-fallback implementation notes from router-visible
    Bilibili, YouTube, and nHentai source descriptions.
  - Kept SearXNG as the generic subagent implementation path in code and ICD,
    but not as router prompt knowledge.
  - Removed the evaluator retry/cap field from the evaluator LLM payload;
    deterministic code still owns the retry cap.
  - Changed the finalizer LLM payload to receive the clean
    `evaluator_feedback` string instead of evaluator message wrapper metadata.
  - Added focused failing tests first, then implemented the prompt/payload
    cleanup.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py::test_web_agent3_router_source_text_omits_execution_details tests\test_web_agent3.py::test_web_agent3_evaluator_continues_with_feedback tests\test_web_agent3.py::test_web_agent3_finalizer_payload_uses_clean_feedback -q`: failed before implementation, then 3 passed after implementation.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\__init__.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\youtube.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py -q`: 20 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py -q`: 26 passed.
  - Static check `rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|httpx|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`: matches only `searxng_tools.py`.
  - Static check `rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name|provider metadata" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py`: no matches.
- 2026-05-27 apple-to-apple side-by-side live LLM comparison:
  - Replaced the comparison harness boundary with public helper calls:
    `WebSearchAgent().run(task, context, max_attempts=3)` versus
    `WebAgent3().run(task, context, max_attempts=3)`.
  - Both agents receive the same task/context and the same deterministic
    patched SearXNG fixture backend while real LLM stages remain active.
  - `venv\Scripts\python -m py_compile tests\test_web_agent_comparison_live_llm.py`: passed.
  - `venv\Scripts\python -m pytest -m live_llm tests\test_web_agent_comparison_live_llm.py --collect-only -q`: 10 tests collected.
  - Ran all 10 live comparison cases one at a time with
    `venv\Scripts\python -m pytest tests\test_web_agent_comparison_live_llm.py::<case> -q -s -m live_llm`: all 10 passed structural assertions and wrote trace artifacts.
  - Debug review:
    `test_artifacts/llm_reviews/web_agent2_vs_web_agent3_public_run_live_llm_review_20260527.md`.
  - Key RCA signals: web_agent3 regresses on YouTube and Bilibili URL tasks
    because placeholder source routing returns `no_search_data`; web_agent3
    also failed to mark stale news as stale. web_agent3 improved the
    no-relevant-info case by returning `resolved=false` with negative evidence.
  - Result: not ready for big-bang cutover without addressing these parity
    gaps or explicitly accepting placeholder-source regressions.
- 2026-05-27 dedicated routing edge-case unit coverage:
  - Added `tests/test_web_agent3_routing.py` to isolate deterministic router
    boundary checks from broader helper tests.
  - Covered malformed action/source normalization, empty query fallback, stop
    query clearing, valid edge-source route preservation for YouTube,
    Bilibili, nHentai, and generic URL reads, invalid route fallback to generic
    search, and executor dispatch without cross-source fallback.
  - `venv\Scripts\python -m py_compile tests\test_web_agent3_routing.py`:
    passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3_routing.py -q`: 4
    passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py tests\test_web_agent3_routing.py -q`: 24 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py tests\test_web_agent3_routing.py -q`: 30 passed.
- 2026-05-27 Stage 3C nHentai planning update:
  - User approved replacing the nHentai placeholder with metadata-only
    official API v2 support.
  - Recorded scope: `read` returns gallery name/title and grouped tags only;
    `search` returns bounded gallery candidates only.
  - Recorded constraints: no download, no picture/page read, no thumbnails, no
    comments, no favorite/account/moderation endpoints, no token/header leak,
    no generic router expansion, and no SearXNG bypass for ordinary web search
    or URL reads.
  - No production code was changed for this planning update.
- 2026-05-27 Stage 3C nHentai implementation:
  - Replaced `subagent/nhentai.py` placeholder with source-local gallery id
    extraction, optional process-environment API token handling, official API
    metadata read, official API search, response compaction, and bounded error
    observations.
  - Kept router/generator contract unchanged: `action`, `source`, and `query`
    only. nHentai id extraction and API parameter shaping live inside the
    nHentai subagent.
  - Updated `src/kazusa_ai_chatbot/rag/web_agent3/README.md` to mark nHentai
    as metadata/search API-backed and Bilibili/YouTube as the remaining
    no-data placeholders.
  - Updated `tests/test_web_agent3.py` placeholder expectations so only
    Bilibili and YouTube remain no-data placeholder sources.
  - Added `tests/test_web_agent3_nhentai.py` with deterministic coverage for
    id extraction, metadata read, bounded search, numeric search lookup,
    missing gallery ids, token header behavior, API 404/429/HTTP/JSON/shape
    errors, and forbidden data leakage.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py tests\test_web_agent3_nhentai.py`:
    passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\providers.py src\kazusa_ai_chatbot\rag\web_agent3\searxng_tools.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\__init__.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\youtube.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py tests\test_web_agent_comparison_live_llm.py`:
    passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3_nhentai.py -q`: 6
    passed.
  - `venv\Scripts\python -m pytest tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py -q`:
    30 passed.
  - `venv\Scripts\python -m pytest tests\test_web_search_agent.py tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py -q`:
    36 passed.
  - Static check `rg "WEB_AGENT3_SEARXNG|WEB_AGENT3_HTTP|url_safety|html_extract" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py`:
    no matches.
  - Static check `rg "httpx|NHENTAI_TOKEN|Authorization" src\kazusa_ai_chatbot\rag\web_agent3`:
    matches only `subagent/nhentai.py`.
  - Static check `rg "NHENTAI_TOKEN|Authorization" tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py`:
    matches only `tests/test_web_agent3_nhentai.py`.
  - Static check `rg "mcp-searxng__searxng_web_search|mcp-searxng__web_url_read" src\kazusa_ai_chatbot\rag\web_agent3`:
    matches only `searxng_tools.py`.
  - Static check `rg "# 输入格式|# Input Format|locator|locator_type|provider_history|selected_provider_name|executed_provider_name" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py`:
    no matches.
  - CJK quote safety check `rg '[“”]' src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py tests\test_web_agent3_nhentai.py`:
    no matches.
  - Broad exception check `rg "except Exception|except:" src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`:
    no matches.
  - Independent Stage 3C code review subagent found no blocking findings.
    Residual risk: live nHentai API behavior was not smoke-tested.

## Independent Plan Review

Review date: 2026-05-26.

Inputs reviewed: this active plan, the current `web_search_agent2` contract,
the user-confirmed router contract, project RAG ownership rules, and current
pre-cutover change surface.

Findings:

- No blocker: Stage 3A now encodes the user-confirmed architecture instead of
  the earlier deterministic-router interpretation.
- No blocker: The plan keeps upstream/downstream boundaries stable. The RAG
  initializer still produces semantic `Web-evidence:` work, and downstream
  RAG/cognition still receives text evidence.
- No blocker: The execution surface remains limited to `web_agent3` package,
  focused web-agent tests, comparison tests, and this plan until cutover.
- Non-blocking risk: The current working-tree code still reflects the older
  metadata-heavy implementation. Stage 3A must be completed before rerunning
  live comparison or attempting cutover.
- Non-blocking risk: The static grep allows comparison-fixture metadata if
  needed. Before final cutover, runtime code should have no dependency on those
  metadata fields.

Review result: approved for Stage 3A implementation. Do not proceed to live
LLM comparison or big-bang transition until Stage 3A focused verification and
static checks pass.

## Acceptance Criteria

- `WebAgent3.run` keeps the same public contract as `WebSearchAgent.run`.
- `web_agent3` generic search/read execution uses the existing SearXNG MCP
  facility.
- No direct HTTP search/fetch, URL safety module, or HTML extraction module is
  introduced for generic web search or URL read.
- nHentai is the only approved direct provider HTTP client in pre-cutover
  `web_agent3`, and it reads only the optional `NHENTAI_TOKEN` environment
  variable inside `subagent/nhentai.py`.
- Router/generator emits only `action`, `source`, and `query`; prompts omit
  `# 输入格式` / `# Input Format`.
- Router/generator source descriptions do not expose transport details,
  placeholder execution details, or provider metadata.
- Router normalization and executor dispatch are covered for edge cases around
  malformed LLM output, stop actions, source-specific IDs, site URLs, and
  placeholder-source boundaries.
- Evaluator/finalizer LLM payloads contain only fields each stage uses for its
  semantic decision.
- Bilibili and YouTube placeholder source adapters return no search data
  without calling generic SearXNG and carry the fixed FIXME marker.
- nHentai `read` returns gallery title/name and grouped tags only; nHentai
  `search` returns bounded gallery candidates only; neither path returns
  images, download URLs, thumbnails, comments, favorite/account data, raw
  headers, or tokens.
- Prompt content follows project language and cache-prefix rules.
- Focused deterministic tests pass.
- The ten real LLM comparison cases are rerun one at a time and summarized in a
  human-authored debug-LLM report before any cutover.
