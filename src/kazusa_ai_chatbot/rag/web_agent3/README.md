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
| `searxng_tools.py` | LangChain tool wrappers for direct SearXNG search and direct URL reads. |
| `direct_searxng.py` | Direct SearXNG JSON search client and result formatter. |
| `url_reader.py` | Process-local HTTP(S) URL reader and text extractor. |
| `providers.py` | Thin source decision executor used by the graph executor. |
| `subagent/__init__.py` | Auto-discovery and validation for source subagent modules. |
| `subagent/web_read.py` | Direct URL-read source, always available. |
| `subagent/web_search.py` | Direct search source, available when `SEARXNG_URL` is configured. |
| `subagent/nhentai.py` | nHentai API v2 metadata/search source, available when `NHENTAI_TOKEN` is configured. |
| `contracts.py` | Minimal router decision and test/comparison data contracts. |

## Direct Web Facility Rule

Ordinary webpage search is exposed through `web_search` when the direct search
endpoint is configured by `SEARXNG_URL`. If `SEARXNG_URL` is empty,
`web_search` is not registered and the router does not see search as an
available source.

URL-read execution is process-local HTTP(S) fetching. It does not require
SearXNG or MCP. Localhost, loopback, private LAN addresses, and intranet
hostnames are allowed by default because the reader observes resources from
the Kazusa process perspective. The reader owns browser-like request headers,
timeouts, response-size caps, deterministic text extraction, and bounded error
strings.

The URL reader always sends browser-navigation headers, advertises only
compression encodings the local HTTP stack can decode, and keeps response
cookies in process memory for later reads. It also detects common HTTP
anti-bot challenge surfaces such as Cloudflare, DataDome, Akamai, PerimeterX,
and generic CAPTCHA pages before returning a generic HTTP-status error. It
does not execute JavaScript, solve CAPTCHA, or impersonate browser TLS
fingerprints; pages that require those mechanisms return a bounded blocked
observation for the graph.

Source-specific metadata/search is owned by the selected source subagent.
Currently, only `nhentai` has an approved direct provider API path, limited to
gallery metadata reads and gallery search through its own subagent. If
`NHENTAI_TOKEN` is empty, `nhentai` is not registered.

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
    "source": "web_search",
    "query": "local tool router demo web agent architecture"
}
```

Allowed actions are `search`, `read`, and `stop`. Allowed sources are the
enabled source modules discovered from `subagent/*.py`. The executor dispatches
only by `source` and `action`; it does not reinterpret `query`. `stop` is
handled by the graph executor before source dispatch.

`query` is passed unchanged to the selected source subagent. Source-specific
ID extraction, API parameter building, credential use, and tool variants are
subagent responsibilities. `nhentai` implements deterministic gallery-id
extraction and API parameter building.

## Source Subagents

The source dispatcher selects from the enabled final roster:

- `web_read`: always available for direct HTTP(S) URL reads.
- `web_search`: available when `SEARXNG_URL` is configured.
- `nhentai`: available when `NHENTAI_TOKEN` is configured.

The source subagent roster and prompt-facing descriptions are auto-discovered
from `subagent/*.py`. Each source module exposes `SOURCE`, `DESCRIPTION`,
`SUPPORTED_ACTIONS`, and `execute(...)`. Configuration-dependent source modules
also expose `is_enabled()`. `DESCRIPTION` includes source-local `query`
generation rules for the router prompt.

`web_read` uses direct URL reads. `web_search` uses the configured direct
search endpoint. `nhentai` uses the official API v2 for metadata-only gallery
reads and bounded gallery searches. The nHentai subagent imports
`NHENTAI_TOKEN` and `NHENTAI_SOURCE_ENABLED` from `config.py`; it must not put
credentials, headers, image URLs, download URLs, comments, favorite state, or
account data into observations.

## Create New Subagent

New source subagents follow this interface guide:

- Create one source module under `subagent/`.
- Expose `SOURCE`, `DESCRIPTION`, and `execute(decision)`.
- Keep source-specific execution inside the source module.
- Keep target parsing inside the source module.
- Keep API parameter construction inside the source module.
- Keep request execution inside the source module.
- Keep result compaction inside the source module.
- Keep source limits inside the source module.
- Keep source error observations inside the source module.
- Keep stable provider constants inside the source module.
- Place user-specific configuration in `config.py`.
- Place deployment-specific configuration in `config.py`.
- Read configuration by importing constants from `kazusa_ai_chatbot.config`.
- Represent configuration-dependent availability with `is_enabled()`.
- Register available subagents through package discovery.
- Keep `DESCRIPTION` focused on user-visible capability and query-shaping
  guidance.
- Return prompt-safe observations with bounded source evidence.
- Add deterministic tests for configuration-dependent registration states.
- Add deterministic tests for source execution with injected configuration.
- Add deterministic tests for prompt-safe observations.
- Update this ICD when the source interface changes.
- Update `docs/HOWTO.md` when the source adds user-specific or
  deployment-specific configuration.

## Current Verification

Focused deterministic tests must cover:

- direct SearXNG JSON search and direct process-local URL reads.
- `WebAgent3.run` public contract parity.
- strict router output parsing into `action`, `source`, and `query`.
- routing edge cases for malformed actions/sources, empty queries, stop
  decisions, source-specific IDs, site URLs, and executor dispatch boundaries.
- query pass-through from executor to source subagents.
- source subagent discovery from per-source modules under `subagent/`.
- source-local query generation rules rendered from subagent descriptions.
- configuration-dependent source registration for `web_search` and `nhentai`.
- graph-local stop handling before source dispatch.
- source/action normalization for `web_read`, `web_search`, and `nhentai`.
- nHentai `read` returns only compact title/name and grouped tags.
- nHentai `search` returns bounded gallery candidates without image, download,
  comment, favorite, header, or token data.
- generator/evaluator prompt payload placement for `reference_time`.
- finalizer comparison helper shape used by live LLM reports.
