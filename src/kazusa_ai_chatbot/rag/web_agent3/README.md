# web_agent3 ICD

`kazusa_ai_chatbot.rag.web_agent3` is the candidate replacement for
`web_search_agent2`. It keeps the same RAG helper input and output contract
while adding a small internal source router.

## Public Contract

```python
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3

await WebAgent3().run(task: str, context: dict, max_attempts: int = 3)
```

The returned shape matches `web_search_agent2`:

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

No caller supplies provider names, credentials, URLs outside the task/context,
or new RAG supervisor fields.

## Internal Modules

| Module | Ownership |
|---|---|
| `agent.py` | LangGraph router/generator, executor, evaluator, finalizer, and `WebAgent3` wrapper. |
| `searxng_tools.py` | Existing SearXNG MCP facility calls for search and URL reads. |
| `providers.py` | Thin source decision executor used by the graph executor. |
| `subagent/__init__.py` | Auto-discovery and validation for source subagent modules. |
| `subagent/generic.py` | Generic search/read subagent backed by the existing SearXNG facility. |
| `subagent/bilibili.py` | Bilibili placeholder subagent. |
| `subagent/youtube.py` | YouTube placeholder subagent. |
| `subagent/nhentai.py` | nHentai placeholder subagent. |
| `contracts.py` | Minimal router decision and test/comparison data contracts. |

## SearXNG Facility Rule

All web search and URL-read execution goes through the existing SearXNG MCP
facility. `web_agent3` does not perform direct HTTP search, direct URL fetch,
SSRF filtering, or custom HTML extraction. Those concerns remain owned by the
configured SearXNG/MCP server.

## Prompt Rule

Prompts are Chinese-first RAG evidence prompts. Stable role, policy, generation
procedure, and output contracts stay in `SystemMessage`. Runtime-varying fields
such as `task`, projected `context`, `reference_time`, tool history, and
evaluator feedback stay in `HumanMessage` JSON so early prompt prefixes remain
stable for local LLM prefix caching.

## Internal Flow

```text
router/generator -> executor -> source subagent -> evaluator -> loop/finalizer
```

The router/generator is the first LLM stage. It returns exactly:

```json
{
    "action": "search",
    "source": "generic",
    "query": "local tool router demo web agent architecture"
}
```

Allowed actions are `search`, `read`, and `stop`. Allowed sources are
`generic`, `bilibili`, `youtube`, and `nhentai`. The executor dispatches only
by `source` and `action`; it does not reinterpret `query`.

`query` is passed unchanged to the selected source subagent. Source-specific
ID extraction, API parameter building, credentials, and local/MCP tool variants
are intentionally deferred to future subagent work.

## Source Subagents

The source dispatcher selects one of:

- `generic`
- `bilibili`
- `youtube`
- `nhentai`

The source subagent roster and prompt-facing descriptions are auto-discovered
from `subagent/*.py`. Each source module exposes `SOURCE`, `DESCRIPTION`, and
`execute(...)`. `DESCRIPTION` includes source-local `query` generation rules
for the router prompt. `generic` uses the existing SearXNG facility directly.
The other source adapters prove the dispatch point for future provider APIs. In
this stage they return an explicit `no_search_data` placeholder instead of
falling back to generic search, and carry:

```text
FIXME(web_agent3): replace no-search-data placeholder with provider API client in a future approved plan.
```

Real Bilibili, YouTube, nHentai, local-tool variants, MCP-backed variants,
credentials, rate limits, and source-specific parsers require a later approved
plan.

## Current Verification

Focused deterministic tests must cover:

- SearXNG MCP delegation for search and URL reads.
- `WebAgent3.run` public contract parity.
- strict router output parsing into `action`, `source`, and `query`.
- routing edge cases for malformed actions/sources, empty queries, stop
  decisions, source-specific IDs, site URLs, and executor dispatch boundaries.
- query pass-through from executor to source subagents.
- source subagent discovery from per-source modules under `subagent/`.
- source-local query generation rules rendered from subagent descriptions.
- placeholder source adapters return no search data without calling SearXNG.
- generator/evaluator prompt payload placement for `reference_time`.
- finalizer comparison helper shape used by live LLM reports.
