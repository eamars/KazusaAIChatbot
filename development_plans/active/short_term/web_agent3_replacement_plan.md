# web_agent3 Replacement Plan

Status: draft
Class: medium
Created: 2026-05-25
Owner: Codex

## Summary

Replace the current MCP-backed `web_search_agent2` helper with `web_agent3`, a
local-tool-first web helper owned entirely inside the web agent boundary. The
outer RAG/cognition contract remains the same: the supervisor can ask for web
evidence, and the helper returns bounded evidence text for projection and later
cognition stages.

The first implementation step lays down the internal router architecture and
local generic web tools only. Provider-specific API clients for Bilibili,
YouTube, nHentai, or any MCP-backed provider are extension points, not part of
this plan.

## Context

The current helper is implemented in
`src/kazusa_ai_chatbot/rag/web_search_agent.py` and is registered externally as
`web_search_agent2` from
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`.

Current behavior:

- `web_search_agent2` exposes two LangChain tools: `web_search` and
  `web_url_read`.
- Those tools delegate to `mcp-searxng` via `mcp_manager.call_tool`.
- The helper uses generator, evaluator, and finalizer LLM stages.
- The public helper result is a dictionary whose `result` field is evidence
  text.
- `live_context_agent` delegates external live lookup to the same helper.
- RAG projection treats web evidence as external evidence, not persona or final
  stance.

The requested target is:

- New service/helper name: `web_agent3`.
- Specialized agents live within the web agent, not beside the RAG supervisor.
- Locally hosted MCP services are not required for the first replacement.
- Local Python tool calls are preferred for maintainability.
- Credentials, endpoint URLs, and provider keys can be read from local
  environment variables when provider clients are added.
- The helper must support both search-style tasks and URL-provided summary or
  description extraction.

## Mandatory Skills

Before implementation, the execution agent must load and follow:

- `.agents/skills/py-style/SKILL.md` before editing Python.
- `.agents/skills/test-style-and-execution/SKILL.md` before adding, changing,
  or running tests.
- `.agents/skills/development-plan/SKILL.md` before changing this plan status
  or executing it.
- `.agents/skills/local-llm-architecture/SKILL.md` before changing LLM prompts,
  routing, or tool context budgets.

## Mandatory Rules

- Do not read `.env`.
- Do not change cognition, dialog, memory, persistence, adapters, scheduler,
  reflection, or database code.
- Do not add a new MCP service.
- Do not retain a runtime compatibility path from `web_agent3` back to
  `web_search_agent2`.
- Do not register both `web_search_agent2` and `web_agent3` at the same time.
- Do not add provider-specific Bilibili, YouTube, or nHentai clients in this
  plan.
- Do not add new third-party dependencies.
- Use `apply_patch` for manual edits.
- Use `venv\Scripts\python` for Python and pytest commands.

## Must Do

- Rename the current web helper surface from `web_search_agent2` to
  `web_agent3`.
- Replace MCP tool calls with local Python tool calls inside the web helper.
- Keep specialized routing ownership inside the web helper.
- Add a provider/tool registry seam inside `web_agent3` so future local or MCP
  providers can be added without changing the outer RAG supervisor.
- Implement generic local search through a direct SearXNG-compatible HTTP JSON
  endpoint when configured.
- Implement local URL fetch and bounded metadata/content extraction with
  `httpx` and the Python standard library.
- Preserve the public helper return shape expected by the RAG supervisor:
  `{"resolved": bool, "result": str, "attempts": int, "cache": ...}`.
- Keep evidence bounded, source-attributed, and projection-friendly.
- Update tests that assert the old agent name or MCP delegation.
- Update local docs that mention the old helper name or MCP-only web search
  setup.

## Deferred

- Bilibili API client.
- YouTube Data API client.
- nHentai API client.
- Any MCP-backed provider implementation.
- Provider credential validation beyond reading environment variables inside
  the web helper.
- New global config objects.
- New pyproject dependencies.
- Any changes to outer planner semantics beyond the agent rename.
- Any change to the cognition prompt, dialog prompt, or final response style.

## Cutover Policy

Overall strategy: `bigbang`.

Cutover behavior:

- Replace `web_search_agent2` with `web_agent3` in one implementation pass.
- Remove the old MCP-backed web helper module after the rename.
- Update dispatch, projection, live-context delegation, tests, and docs in the
  same change.
- There is no runtime fallback to `web_search_agent2`.
- Existing archived plans and historical docs can keep old names as historical
  records.

## Target State

The target runtime shape is:

```text
RAG supervisor
  -> web_agent3
       -> WebRequest classifier
       -> provider registry
       -> generic local web provider
            -> local search tool
            -> local URL fetch tool
       -> bounded evidence finalizer
  -> projection as external evidence
  -> cognition consumes evidence
```

The RAG supervisor still asks for a web helper by name. It does not learn site
APIs, credentials, provider routing rules, or provider-specific tool schemas.

`web_agent3` owns:

- URL extraction.
- Search-vs-URL task shaping.
- Provider eligibility.
- Local HTTP execution.
- Tool result normalization.
- Evidence summarization.
- Provider extension registry.

## Design Decisions

### Agent Boundary

`web_agent3` is the only new web helper exposed to the RAG supervisor. All
specialized provider logic remains under the web helper boundary. The outer RAG
dispatch table and projection code only need to know the helper name and the
existing result contract.

### Local-Tool-First Execution

The initial provider is `generic_local_web`. It uses direct HTTP calls through
`httpx`:

- Search calls a SearXNG-compatible JSON endpoint when
  `WEB_AGENT3_SEARXNG_BASE_URL` is set.
- URL fetch calls the provided URL directly.
- HTML title and metadata extraction uses a small `html.parser.HTMLParser`
  subclass inside the web helper.
- Text is clipped before it enters any LLM prompt.

The implementation does not call `mcp_manager`, `mcp-searxng`,
`searxng_web_search`, or `web_url_read`.

### Extension Registry

`web_agent3` contains an internal provider registry with a narrow protocol:

```python
class WebProvider(Protocol):
    name: str
    backend_kind: Literal["local", "mcp"]
    supported_domains: tuple[str, ...]

    async def can_handle(self, request: WebRequest) -> ProviderMatch: ...
    async def search(self, request: WebRequest) -> WebToolResult: ...
    async def fetch_url(self, request: WebRequest) -> WebToolResult: ...
```

Only `backend_kind="local"` providers are registered in this plan. The
`"mcp"` backend kind is reserved in the data contract so a later approved plan
can add an MCP adapter without changing the outer supervisor contract.

### Provider Selection

Selection is deterministic in this plan:

1. Extract URLs from the task.
2. If a URL is present, choose the best provider by normalized domain.
3. If no provider claims the domain, use `generic_local_web.fetch_url`.
4. If no URL is present, use `generic_local_web.search`.
5. If the configured search endpoint is absent, return unresolved with a
   clear missing-context reason.

No new LLM router is added in this plan.

### Evidence Finalization

The finalizer remains inside `web_agent3` and produces compact external
evidence text. It should include:

- What was looked up.
- Source URL or search result URL when available.
- Title or description when available.
- Short summary of relevant page/search content.
- Clear unresolved reason when the tool could not retrieve evidence.

The finalizer must not claim facts beyond retrieved tool output.

### Public Contract

The helper public return shape remains supervisor-compatible:

```python
{
    "resolved": bool,
    "result": str,
    "attempts": int,
    "cache": {
        "key": None,
        "hit": False,
        "reason": "agent_not_cacheable",
    },
}
```

Internal typed objects can be richer, but the projection-facing `result` field
stays text in this plan. That keeps the replacement focused and avoids changing
cognition inputs.

### Environment Handling

The implementation reads only the specific environment values it needs from
inside `web_agent3`; it does not add central config fields in this plan.

Reserved environment names:

- `WEB_AGENT3_SEARXNG_BASE_URL`
- `WEB_AGENT3_USER_AGENT`
- `WEB_AGENT3_HTTP_TIMEOUT_SECONDS`

Provider API keys for Bilibili, YouTube, and nHentai are not read until those
providers are implemented under later approved plans.

## Contracts And Data Shapes

### `WebRequest`

Fields:

- `task: str`
- `context: Mapping[str, Any]`
- `urls: tuple[str, ...]`
- `query: str`
- `mode: Literal["search", "fetch_url"]`

The request is derived inside `web_agent3` from the existing helper inputs.

### `ProviderMatch`

Fields:

- `provider_name: str`
- `score: float`
- `reason: str`

The score is deterministic and only ranks provider eligibility. It does not
represent evidence confidence.

### `WebToolResult`

Fields:

- `resolved: bool`
- `provider_name: str`
- `backend_kind: Literal["local", "mcp"]`
- `operation: Literal["search", "fetch_url"]`
- `query: str | None`
- `url: str | None`
- `title: str | None`
- `description: str | None`
- `content: str`
- `items: list[WebSearchItem]`
- `error: str | None`

`content` and every item snippet must be clipped before finalization.

### `WebSearchItem`

Fields:

- `title: str`
- `url: str`
- `snippet: str`
- `source: str | None`

## LLM Call And Context Budget

Current `web_search_agent2` can spend multiple LLM calls through generator,
evaluator, and finalizer stages. The replacement reduces this.

`web_agent3` budget:

- 0 LLM calls for request classification and provider selection.
- 1 local HTTP call for direct URL fetch.
- 1 local HTTP call for generic SearXNG search.
- 1 finalizer LLM call to convert tool output into compact evidence text.
- No evaluator LLM loop.
- No repair loop around malformed finalizer output.

Prompt context limits:

- Clip fetched page text before finalization.
- Clip each search item snippet before finalization.
- Limit the number of search items passed to the finalizer.
- Include source URLs explicitly so projection and cognition can reason about
  evidence provenance.

If the finalizer response is missing the required evidence field, the helper
returns unresolved with a structured error reason instead of inventing
evidence.

## Change Surface

Runtime files:

- Rename or replace
  `src/kazusa_ai_chatbot/rag/web_search_agent.py` with
  `src/kazusa_ai_chatbot/rag/web_agent3.py`.
- Update `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py` only
  for import, registry key, prefix dispatch, prompt tool name, and union type
  rename.
- Update `src/kazusa_ai_chatbot/rag/live_context_agent.py` only for import,
  helper construction, worker payload key, and `primary_worker` rename.
- Update
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py` only for
  the web agent name check.

Tests:

- Rename or replace `tests/test_web_search_agent.py` with
  `tests/test_web_agent3.py`.
- Update existing tests and fixtures only where they assert
  `web_search_agent2`, `web_search_agent.py`, MCP web-tool names, or MCP
  delegation.
- Add deterministic tests for local URL fetch, local SearXNG search response
  normalization, provider registry selection, unresolved missing endpoint
  behavior, and public helper return shape.
- Add one live-LLM marked finalizer test using patched tool output if the test
  suite already has the required live-LLM environment.

Documentation:

- Update `src/kazusa_ai_chatbot/rag/README.md` for the helper name and local
  tool ownership.
- Update `docs/HOWTO.md` for `web_agent3` local web settings.

Forbidden runtime changes:

- `src/kazusa_ai_chatbot/mcp_client.py`
- `src/kazusa_ai_chatbot/config.py`
- Adapter modules
- Cognition modules
- Dialog modules
- Persistence and database modules
- Scheduler and reflection modules
- `pyproject.toml`

## Overdesign Guardrail

This plan lays down the smallest useful router shape:

- One exposed agent name.
- One internal provider protocol.
- One generic local provider.
- One finalizer.
- No provider package split.
- No plugin loader.
- No MCP adapter implementation.
- No external planner changes.

Provider-specific clients require later plans with their own tests, credentials,
rate-limit rules, and content-policy boundaries.

## Agent Autonomy Boundaries

During execution, the agent may:

- Rename the old web helper to `web_agent3`.
- Replace MCP calls with local `httpx` calls inside the web helper.
- Introduce internal dataclasses and protocols inside `web_agent3`.
- Update direct rename/wiring references required for the agent name change.
- Update tests and documentation listed in the change surface.

During execution, the agent must not:

- Add specialized provider clients.
- Add a new service process.
- Add MCP server configuration.
- Change global config objects.
- Change RAG supervisor semantics beyond the helper name.
- Change cognition or dialog prompts.
- Add caching behavior.
- Change database schemas or stored data.

## Implementation Order

1. Confirm current state.
   - Run `git status --short`.
   - Re-read this plan.
   - Re-read `README.md`, `docs/HOWTO.md`,
     `src/kazusa_ai_chatbot/rag/README.md`, and directly affected source/tests.

2. Apply the rename boundary.
   - Rename the helper module to `web_agent3`.
   - Rename the public class to `WebAgent3`.
   - Keep public helper return shape unchanged.

3. Replace tool execution.
   - Remove LangChain MCP web tools from the helper.
   - Add internal request, provider, and tool-result dataclasses.
   - Add direct local SearXNG search execution.
   - Add direct URL fetch and metadata extraction.
   - Clip tool output before finalization.

4. Simplify agent execution.
   - Replace generator/evaluator graph behavior with deterministic request
     shaping, provider selection, local tool execution, and one finalizer call.
   - Return unresolved results for missing search endpoint, HTTP failures,
     invalid URLs, or malformed finalizer output.

5. Update rename-only wiring.
   - Dispatch registry and prefix table use `web_agent3`.
   - Live-context delegation records `web_agent3`.
   - Projection recognizes `web_agent3` as external web evidence.

6. Update tests.
   - Replace MCP delegation tests with local-tool tests.
   - Update agent-name assertions.
   - Add provider-registry and unresolved-path tests.
   - Add live-LLM finalizer coverage only under the existing live-LLM marker
     contract.

7. Update docs.
   - Replace MCP-only web-search setup with `web_agent3` local-tool setup.
   - Document reserved provider extension boundary without documenting
     unimplemented providers as available features.

## Execution Model

Use parent-led execution. The parent agent owns the plan, status updates, and
final sign-off.

Native subagents may be used for isolated review tasks:

- One subagent can inspect rename fallout and `rg` results.
- One subagent can review test coverage against this plan.

Subagents must not edit files unless the parent delegates an explicit file list
and scope.

## Progress Checklist

- [ ] Status changed from `draft` to `approved` before implementation.
- [ ] Current git state recorded.
- [ ] `py-style` skill loaded before Python edits.
- [ ] `test-style-and-execution` skill loaded before test edits or test runs.
- [ ] Old MCP tool calls removed from the replacement helper.
- [ ] `web_agent3` internal provider registry implemented.
- [ ] Generic local search implemented.
- [ ] Generic local URL fetch implemented.
- [ ] Public helper return shape preserved.
- [ ] Dispatch rename completed.
- [ ] Live-context rename completed.
- [ ] Projection rename completed.
- [ ] Tests renamed and updated.
- [ ] Documentation updated.
- [ ] Deterministic tests passed.
- [ ] Live-LLM finalizer test run one case at a time when environment is
  available.
- [ ] Independent code review completed.
- [ ] Plan status changed to `completed` only after verification evidence is
  recorded.

## Verification

Run after implementation:

```powershell
venv\Scripts\python -m pytest tests/test_web_agent3.py -q
```

```powershell
venv\Scripts\python -m pytest tests/test_rag_projection.py tests/test_persona_supervisor2_rag2_integration.py tests/test_rag_phase3_supervisor_integration.py -q
```

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

Run the live-LLM finalizer test only when the live-LLM test environment is
available:

```powershell
venv\Scripts\python -m pytest tests/test_web_agent3_live_llm.py::test_live_web_agent3_finalizer_compacts_tool_output -q -s -m live_llm
```

Search checks:

```powershell
rg "web_search_agent2|web_search_agent.py|WebSearchAgent" src tests docs
```

Expected result: no active runtime or test references remain. Historical
development plans and archived docs are outside this check.

```powershell
rg "mcp-searxng|searxng_web_search|web_url_read|mcp_manager" src/kazusa_ai_chatbot/rag/web_agent3.py tests/test_web_agent3.py
```

Expected result: no matches.

## Independent Code Review

After implementation and deterministic tests pass, request an independent code
review focused on:

- Scope containment against this plan.
- No accidental cognition, dialog, database, adapter, or scheduler changes.
- No MCP runtime path in `web_agent3`.
- Public helper result compatibility.
- Evidence clipping and source attribution.
- Test coverage for unresolved network/configuration paths.

## Acceptance Criteria

- `web_agent3` fully replaces `web_search_agent2` in active runtime and tests.
- The old MCP-backed helper is removed from the active code path.
- The RAG supervisor can still dispatch `Web-evidence:` and `Web-search:`
  requests.
- URL-provided tasks can return bounded description or summary evidence.
- Search tasks use a direct local SearXNG-compatible endpoint when configured.
- Missing local search configuration returns a clear unresolved helper result.
- The internal provider registry reserves both local and MCP backend kinds, but
  only local generic tooling is registered.
- No specialized provider client is implemented in this plan.
- Deterministic tests pass.
- Live-LLM finalizer behavior is inspected one case at a time when the
  environment is available.
- No forbidden runtime files are changed.
