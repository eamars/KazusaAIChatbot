# web_agent3 Replacement Plan

Status: draft
Class: large
Created: 2026-05-25
Last updated: 2026-05-26
Owner: Codex

## Summary

- Goal: build `web_agent3` as a local-tool-first replacement for
  `web_search_agent2`, then cut active RAG wiring over in one big-bang rename.
- Contract rule: `web_agent3` must keep the same helper input/output contract
  as `web_search_agent2`; only the active agent name changes.
- Execution order: implement and verify all `web_agent3` functions first while
  `web_search_agent2` remains the active runtime helper, then perform the
  big-bang transition.
- Architecture: provider routing lives inside `web_agent3`; the RAG supervisor
  still sees one web helper and one text evidence result.
- Dummy handlers: Bilibili, YouTube, and nHentai handlers must exist now to
  prove router capability, but they must fall back to the generic SearXNG/local
  handler with this fixed marker:
  `FIXME(web_agent3): replace generic fallback with provider API client in a future approved plan.`
- Status: draft. Do not implement until approved.

## Context

`src/kazusa_ai_chatbot/rag/web_search_agent.py` currently defines
`WebSearchAgent`, registered as `web_search_agent2` in the RAG dispatcher. It
delegates search and URL reads through MCP SearXNG tools, then returns evidence
text through the `BaseRAGHelperAgent` contract.

The requested target keeps the RAG/cognition boundary stable:

- Same `run(task, context, max_attempts=3)` input shape.
- Same dictionary output shape.
- Same downstream projection expectation: `result` is evidence text.
- No new supervisor, cognition, dialog, adapter, persistence, or database
  contract.

The change replaces only the web helper implementation and active helper name.

## Mandatory Skills

- `.agents/skills/development-plan/SKILL.md`: load before plan lifecycle
  changes, execution, progress updates, or sign-off.
- `.agents/skills/py-style/SKILL.md`: load before editing Python.
- `.agents/skills/test-style-and-execution/SKILL.md`: load before editing or
  running tests.
- `.agents/skills/local-llm-architecture/SKILL.md`: load before changing
  prompts, LLM calls, routing, or context budgets.

## Mandatory Rules

- Do not read `.env`.
- Do not change cognition, dialog, memory, persistence, adapters, scheduler,
  reflection, database code, `config.py`, `mcp_client.py`, or `pyproject.toml`.
- Do not add a service process, dependency, global config object, or MCP server.
- Do not call `mcp_manager`, `mcp-searxng`, `searxng_web_search`, or
  `web_url_read` from `web_agent3`.
- Do not keep a runtime compatibility path from `web_agent3` to
  `web_search_agent2`.
- Temporary development-time coexistence of `web_agent3.py` and
  `web_search_agent.py` is allowed only before the cutover. Active runtime
  registration must never expose both helpers.
- Establish focused tests before production implementation.
- Implement and verify all `web_agent3` functions before updating dispatcher,
  projection, live-context wiring, or deleting `web_search_agent.py`.
- Use `apply_patch` for manual edits and `venv\Scripts\python` for Python and
  pytest commands.
- After context compaction or major checklist sign-off, reread this plan before
  continuing.
- Run independent code review and record evidence before completion.

## Must Do

- Create `src/kazusa_ai_chatbot/rag/web_agent3.py` with `WebAgent3`.
- Preserve the exact external helper contract listed in
  `Contracts And Data Shapes`.
- Implement local generic search through a SearXNG-compatible HTTP JSON
  endpoint.
- Implement local URL fetch, URL safety validation, metadata extraction, and
  evidence finalization.
- Implement internal provider routing and dummy Bilibili, YouTube, and nHentai
  handlers.
- Make dummy handlers match their domains, record that they were selected, and
  fall back to `generic_local_web` with the fixed `FIXME(web_agent3)` marker.
- Verify focused `web_agent3` function tests before the big-bang transition.
- After focused verification, replace active `web_search_agent2` wiring with
  `web_agent3`, delete `web_search_agent.py`, and update active tests/docs.

## Deferred

- Real Bilibili API client.
- Real YouTube Data API client.
- Real nHentai API client.
- MCP-backed provider implementation.
- Provider credentials and credential validation.
- Caching.
- Any outer RAG planner, cognition, dialog, adapter, persistence, scheduler, or
  database redesign.

## Cutover Policy

Overall strategy: `bigbang`.

| Area | Policy | Instruction |
|---|---|---|
| Function implementation | staged before cutover | Build and verify `web_agent3.py` while old runtime wiring remains active. |
| Runtime registration | bigbang | Replace active `web_search_agent2` references with `web_agent3` in one wiring pass. |
| Old helper | bigbang | Delete `web_search_agent.py` after `web_agent3` focused tests pass and wiring is updated. |
| Dummy handlers | explicit temporary fallback | Current dummy providers fall back to `generic_local_web`; this is allowed by user instruction and must carry the fixed FIXME marker. |
| Tests/docs | bigbang after focused verification | Rename active expectations from `web_search_agent2` to `web_agent3`. |

No released runtime state may register or dispatch both `web_search_agent2` and
`web_agent3`.

## Target State

```text
RAG supervisor
  -> web_agent3.run(task, context, max_attempts=3)
       -> deterministic request shaping
       -> provider registry
          -> bilibili_dummy_provider -> generic_local_web
          -> youtube_dummy_provider  -> generic_local_web
          -> nhentai_dummy_provider  -> generic_local_web
          -> generic_local_web
       -> local search or validated URL fetch
       -> one evidence finalizer
  -> same text result contract as web_search_agent2
```

The supervisor does not learn provider APIs, provider credentials, or provider
tool schemas.

## Design Decisions

- `web_agent3` keeps the same external helper input/output contract as
  `web_search_agent2`.
- Request classification and provider selection are deterministic; no LLM
  router is added.
- The only network executor in this plan is `generic_local_web`.
- Dummy providers are execution-relevant because they prove future routing
  capability now; they do not implement real provider APIs.
- Search uses `WEB_AGENT3_SEARXNG_BASE_URL` from the environment inside
  `web_agent3`; no central config field is added.
- Direct URL fetch must validate task-derived URLs before every request and
  redirect.
- The finalizer returns compact external evidence text and must not invent facts
  beyond tool output.

## Contracts And Data Shapes

### Public Helper Contract

`WebAgent3.run` must accept the same inputs as `WebSearchAgent.run`:

```python
async def run(
    self,
    task: str,
    context: dict[str, Any],
    max_attempts: int = 3,
) -> dict[str, Any]:
```

Contract requirements:

- `task` and `context` are passed by existing RAG callers unchanged.
- `max_attempts` is accepted for compatibility and ignored, matching
  `WebSearchAgent.run`.
- `context["local_time_context"]` is read the same way to build the current
  local timestamp for prompt grounding.
- No caller must provide new input fields.
- `name` changes to `web_agent3`; `cache_name` remains `""`.

Output must match the current BaseRAG helper wrapper shape:

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

`result` must stay a string. Provider routing metadata remains internal or test
visible through focused module helpers; it must not replace the public result
shape.

### Internal Data

- `WebRequest`: task, context, URLs, query, and mode.
- `ProviderMatch`: provider name, score, and reason.
- `WebToolResult`: selected provider, executed provider, operation, query/URL,
  title, description, bounded content, bounded search items, delegation reason,
  missing context, and error.
- `WebSearchItem`: title, URL, snippet, and source.

## Network And Search Contract

Task-derived URL fetch must:

- accept only `http` and `https`;
- reject empty hosts, userinfo, `localhost`, `.local`, and non-global resolved
  IPs;
- reject loopback, private, link-local, multicast, unspecified, reserved, and
  metadata-service addresses;
- disable automatic redirects and manually follow at most three redirects;
- validate every redirect target before requesting it;
- clamp `WEB_AGENT3_HTTP_TIMEOUT_SECONDS` to 1-30 seconds with a 10 second
  default;
- send `WEB_AGENT3_USER_AGENT` or `Kazusa-web-agent3/1.0`;
- read at most 1,000,000 response bytes;
- pass at most 12,000 fetched-content characters to the finalizer;
- accept only textual content types;
- return unresolved for invalid URL, blocked target, DNS failure, HTTP failure,
  timeout, redirect cap, oversized response, or non-textual content.

Generic search must:

- call `GET {WEB_AGENT3_SEARXNG_BASE_URL.rstrip("/")}/search`;
- send `q`, `format=json`, `pageno=1`, `language=auto`, and
  `categories=general`;
- parse a JSON object with `results`;
- keep at most five result items;
- read item `title`, `url`, `content` as snippet, and `engine` as source;
- drop items without `http` or `https` URLs;
- clip title to 200 characters, snippet to 500, and source to 80;
- return unresolved with
  `missing_context=["web_agent3_searxng_base_url"]` when the base URL is absent.

The configured SearXNG base URL is operator-controlled and can be local. It is
not subject to task-derived URL blocking, but timeout and response caps still
apply.

## LLM Call And Context Budget

- Before: `web_search_agent2` can use generator, evaluator, and finalizer LLM
  stages.
- After: `web_agent3` uses deterministic request/provider selection, local tool
  execution, and one finalizer LLM call.
- No evaluator loop and no LLM repair loop.
- Finalizer input caps: five search items, 500 characters per snippet, and
  12,000 fetched-content characters.
- If finalizer output misses the required evidence field, return unresolved
  instead of inventing evidence.

## Change Surface

### Create First

- `src/kazusa_ai_chatbot/rag/web_agent3.py`
- `tests/test_web_agent3.py`

These files are created and verified before runtime wiring changes.

### Big-Bang Transition After Focused Verification

- Delete `src/kazusa_ai_chatbot/rag/web_search_agent.py`.
- Modify rename-only runtime wiring:
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`,
  `src/kazusa_ai_chatbot/rag/live_context_agent.py`, and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`.
- Modify docs:
  `src/kazusa_ai_chatbot/rag/README.md` and `docs/HOWTO.md`.
- Modify active tests/fixtures:
  `tests/fixtures/multi_source_cognition_stage_00_cases.json`,
  `tests/test_persona_supervisor2_rag2_integration.py`,
  `tests/test_quote_aware_rag_sequence.py`,
  `tests/test_rag_initializer_cache2.py`,
  `tests/test_rag_phase3_capability_agents.py`,
  `tests/test_rag_phase3_supervisor_integration.py`,
  `tests/test_rag_phase4_continuation_live_llm.py`,
  `tests/test_rag_projection.py`, and `tests/test_e2e_live_llm.py`.
- Add `tests/test_web_agent3_live_llm.py` only for live-LLM finalizer smoke.

### Keep

`mcp_client.py`, `config.py`, `pyproject.toml`, adapters, cognition, dialog,
persistence, database, scheduler, reflection, and `tests/test_mcp_client.py`.

## Overdesign Guardrail

- Actual problem: the active web helper depends on a separate MCP web service
  and lacks an internal provider router for future site-specific work.
- Minimal change: build `web_agent3` with the same public helper contract,
  generic local search/fetch, and dummy provider routing; cut over after
  focused verification.
- Ownership boundaries: RAG supervisor selects the web helper; `web_agent3`
  owns provider routing and web execution; projection consumes the same text
  evidence; cognition owns stance and response judgment.
- Rejected complexity: real provider APIs, provider credentials, MCP provider
  adapter, plugin loader, LLM router, evaluator loop, repair loop, cache,
  global config, and outer planner redesign.
- Evidence threshold: real provider clients need a later approved plan with API
  contracts, credentials, rate limits, provider tests, and content boundaries.

## Agent Autonomy Boundaries

The execution agent may implement local mechanics inside `web_agent3.py` when
they preserve this plan's contracts. The agent must not add real provider
clients, new services, new dependencies, global config fields, unrelated
refactors, or changes outside the listed files. If implementation reveals that
the same public contract cannot be preserved, stop and report the blocker.

## Implementation Order

1. Baseline.
   - Run `git status --short`.
   - Reread this plan and affected docs/source/tests.
   - Run
     `rg "web_search_agent2|WebSearchAgent|web_search_agent|mcp-searxng|searxng_web_search|web_url_read" src tests docs README.md`.
   - Record git state and grep output.

2. Focused tests before implementation.
   - Create `tests/test_web_agent3.py`.
   - Tests must cover SearXNG search, missing search endpoint, URL metadata
     extraction, blocked local/private/metadata URLs, redirect revalidation,
     dummy provider route-and-fallback behavior, and exact public
     `WebAgent3.run` output contract.
   - Run `venv\Scripts\python -m pytest tests/test_web_agent3.py -q`.
   - Expected before implementation: missing module/symbol or failing new
     contract tests.

3. Implement all `web_agent3` functions.
   - Create `src/kazusa_ai_chatbot/rag/web_agent3.py`.
   - Implement helper contract, local timestamp grounding, request shaping,
     provider registry, dummy handlers, generic SearXNG search, URL validator,
     URL fetch, metadata extraction, and finalizer.
   - Do not touch runtime wiring in this step.
   - Run `venv\Scripts\python -m pytest tests/test_web_agent3.py -q`.
   - Do not proceed to cutover until this passes.

4. Big-bang transition.
   - Delete `web_search_agent.py`.
   - Rename runtime imports, registry key, prefix mapping, prompt-visible helper
     name, projection check, live-context worker payload key, and active test
     expectations from `web_search_agent2` to `web_agent3`.
   - Update docs listed in Change Surface.

5. Integration verification.
   - Run
     `venv\Scripts\python -m pytest tests/test_rag_projection.py tests/test_persona_supervisor2_rag2_integration.py tests/test_rag_phase3_capability_agents.py tests/test_rag_phase3_supervisor_integration.py -q`.
   - Add and run the live-LLM finalizer smoke only when live-LLM environment is
     available.

6. Final verification and review.
   - Run all Verification commands.
   - Start one independent code-review subagent after verification passes.
   - Fix review findings only inside this plan's Change Surface and rerun
     affected checks.

## Execution Model

- Parent owns tests, orchestration, transition wiring, verification, evidence,
  review remediation, lifecycle updates, and sign-off.
- If native subagents are available, one production-code subagent may implement
  `src/kazusa_ai_chatbot/rag/web_agent3.py` only after focused tests exist.
- The parent performs the big-bang transition only after focused tests pass.
- One independent code-review subagent reviews the final diff and evidence.
- If native subagents are unavailable, stop before execution unless the user
  explicitly approves fallback execution.

## Progress Checklist

- [ ] Stage 1 - baseline recorded.
  - Verify: `git status --short` and baseline `rg` completed.
  - Evidence/sign-off: record outputs and sign `<agent/date>`.

- [ ] Stage 2 - focused `web_agent3` tests established.
  - Verify: `venv\Scripts\python -m pytest tests/test_web_agent3.py -q`
    fails for expected missing/new-contract reason.
  - Evidence/sign-off: record failure and sign `<agent/date>`.

- [ ] Stage 3 - all `web_agent3` functions implemented and verified.
  - Verify: `venv\Scripts\python -m pytest tests/test_web_agent3.py -q`
    passes before any runtime cutover.
  - Evidence/sign-off: record changed files and test output, then sign
    `<agent/date>`.

- [ ] Stage 4 - big-bang transition complete.
  - Verify: integration test command in Implementation Order step 5 passes.
  - Evidence/sign-off: record changed wiring/docs/tests and output, then sign
    `<agent/date>`.

- [ ] Stage 5 - final verification and independent code review complete.
  - Verify: all Verification commands pass or live-LLM skip reason is recorded.
  - Evidence/sign-off: record review findings, fixes, reruns, residual risks,
    and sign `<agent/date>`.

## Verification

```powershell
venv\Scripts\python -m pytest tests/test_web_agent3.py -q
```

```powershell
venv\Scripts\python -m pytest tests/test_rag_projection.py tests/test_persona_supervisor2_rag2_integration.py tests/test_rag_phase3_capability_agents.py tests/test_rag_phase3_supervisor_integration.py -q
```

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

Run only when live-LLM environment is available:

```powershell
venv\Scripts\python -m pytest tests/test_web_agent3_live_llm.py::test_live_web_agent3_finalizer_compacts_tool_output -q -s -m live_llm
```

Static checks:

```powershell
rg "web_search_agent2|web_search_agent.py|WebSearchAgent" src tests docs
```

Expected final result: no active runtime, test, or documentation matches.

```powershell
rg "mcp-searxng|searxng_web_search|web_url_read|mcp_manager" src/kazusa_ai_chatbot/rag/web_agent3.py tests/test_web_agent3.py
```

Expected result: no matches.

```powershell
rg "FIXME\\(web_agent3\\): replace generic fallback" src/kazusa_ai_chatbot/rag/web_agent3.py tests/test_web_agent3.py
```

Expected result: matches in dummy-provider code and focused test expectations.

## Independent Plan Review

Before approval or execution, review that the plan preserves the
`web_search_agent2` input/output contract, implements and verifies all
`web_agent3` functions before cutover, keeps dummy providers as generic
fallbacks with FIXME markers, and removes only active old-agent wiring during
the big-bang transition.

## Independent Code Review

After Verification passes, run one independent code review over this plan, the
diff, and execution evidence. Review scope: contract preservation, implementation
order, URL safety, dummy fallback clarity, absence of MCP runtime calls in
`web_agent3`, rename completeness, test coverage, and forbidden-file changes.
Record findings, fixes, reruns, residual risks, and approval status.

## Execution Evidence

Record: baseline git status and grep; focused test expected failure; focused
test pass before cutover; transition diff summary; integration output; static
grep output; deterministic regression output; live-LLM output or skip reason;
independent review result; remediation and rerun output; residual risks.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Public helper contract drift | Preserve exact `run` signature and output shape | Focused contract test |
| URL fetch reaches local/metadata targets | Validate scheme, DNS/IP, and redirects before requests | URL safety focused tests |
| Dummy handlers appear as real provider support | Fixed FIXME marker and delegation reason | Dummy route-and-fallback focused test |
| Cutover before function readiness | Focused tests must pass before wiring changes | Stage 3 gate |
| Rename misses active references | Baseline and final `rg` checks | Static checks |

## Acceptance Criteria

- `WebAgent3.run` accepts the same inputs and returns the same output shape as
  `WebSearchAgent.run`.
- All `web_agent3` functions pass focused tests before runtime wiring changes.
- Active runtime and tests use `web_agent3`, not `web_search_agent2`.
- `web_search_agent.py` is deleted after cutover.
- Generic local search and validated URL fetch work.
- Bilibili, YouTube, and nHentai dummy handlers route and fall back to
  `generic_local_web` with the fixed FIXME marker.
- No real provider API client, MCP web helper path, new service, dependency, or
  forbidden-module change is added.
- Deterministic verification passes and live-LLM finalizer behavior is
  inspected when available.
- Independent code review is completed and recorded.
