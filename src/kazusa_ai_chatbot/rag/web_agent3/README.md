# web_agent3 ICD

`kazusa_ai_chatbot.rag.web_agent3` is the active web evidence helper. It keeps
the standard RAG helper input and output contract while adding a small internal
source router.

## Public Contract

```python
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3

await WebAgent3().run(task: str, context: dict, max_attempts: int = 3)
```

The returned shape is:

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
| `subagent/bilibili.py` | Bilibili subagent with temporary generic web fallback. |
| `subagent/youtube.py` | YouTube subagent with temporary generic web fallback. |
| `subagent/nhentai.py` | nHentai API v2 metadata/search subagent. |
| `contracts.py` | Minimal router decision and test/comparison data contracts. |

## SearXNG Facility Rule

Ordinary webpage search and URL-read execution goes through the existing
SearXNG MCP facility. `web_agent3` does not perform direct HTTP search, direct
URL fetch, SSRF filtering, or custom HTML extraction for generic web work.
Those concerns remain owned by the configured SearXNG/MCP server.

Source-specific metadata/search is owned by the selected source subagent.
Currently, only `nhentai` has an approved direct provider API path, limited to
gallery metadata reads and gallery search through its own subagent.

## Prompt Rule

Prompts are Chinese-first RAG evidence prompts. Stable role, policy, generation
procedure, and output contracts stay in `SystemMessage`. Runtime-varying fields
such as `task`, projected `context`, `reference_time`, tool history, and
evaluator feedback stay in `HumanMessage` JSON so early prompt prefixes remain
stable for local LLM prefix caching.

The web helper uses its own narrow context projection. Public web routing does
not need platform, channel, user, bot, message, or pending resolver ids, so
those identifiers stay out of router, evaluator, and finalizer prompts. The
model receives only semantic query hints such as `original_query`,
`current_slot`, `channel_topic`, and prompt-safe local time context.

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
ID extraction, API parameter building, credential use, and local/MCP tool
variants are subagent responsibilities. In the current stage, only `nhentai`
implements deterministic gallery-id extraction and API parameter building.

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
`nhentai` uses the official API v2 for metadata-only gallery reads and bounded
gallery searches. Bilibili and YouTube prove the dispatch point for future
provider APIs. In this stage each source module temporarily delegates to the
generic SearXNG-backed subagent and carries:

```text
FIXME(web_agent3): replace temporary generic web fallback with a source provider implementation inside this subagent.
```

Real Bilibili, YouTube, local-tool variants, MCP-backed variants, rate-limit
policy, and additional source-specific parsers require a later approved plan.
The nHentai subagent may read its optional API token from the process
environment at execution time and must not put credentials, headers, image
URLs, download URLs, comments, favorite state, or account data into
observations.

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
- Bilibili and YouTube source adapters keep their own modules while using the
  temporary generic web fallback.
- nHentai `read` returns only compact title/name and grouped tags.
- nHentai `search` returns bounded gallery candidates without image, download,
  comment, favorite, header, or token data.
- generator/evaluator prompt payload placement for `reference_time`.
- finalizer comparison helper shape used by live LLM reports.
