# searxng mcp baseline 2026-06-01

This artifact records the current SearXNG MCP behavior before the direct
SearXNG cutover.

- Branch: `searxng-mcp-phaseout`
- MCP endpoint measured: `http://192.168.2.10:4001/mcp`
- SearXNG endpoint observed behind MCP: `http://192.168.2.10:8080`
- MCP connect time: `193.64ms`
- Discovered tools:
  `mcp-searxng__searxng_web_search`,
  `mcp-searxng__web_url_read`
- Examples: `10`
- Successes: `10`
- Failures: `0`
- Latency min: `19.25ms`
- Latency median: `616.93ms`
- Latency max: `1413.68ms`
- Full JSON artifact:
  `development_plans/active/short_term/artifacts/searxng_mcp_baseline_2026-06-01.json`

| Case | Kind | Success | Latency | Chars | First non-empty output line |
|---|---|---:|---:|---:|---|
| `search_01` | search | yes | `1118.15ms` | `3166` | `Title: Search API - SearXNG Documentation (2026.5.31+300695de5)` |
| `search_02` | search | yes | `1257.01ms` | `3072` | `Title: Timeouts - HTTPX` |
| `search_03` | search | yes | `509.17ms` | `3227` | `Title: Responses Overview | OpenAI API Reference` |
| `search_04` | search | yes | `532.93ms` | `3169` | `Title: MRGRD56/textractor-translator - GitHub` |
| `search_05` | search | yes | `1054.77ms` | `2796` | `Title: Kyoto - Wikipedia` |
| `read_01` | read | yes | `92.44ms` | `167` | `# Example Domain` |
| `read_02` | read | yes | `700.93ms` | `6000` | `Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode [Skip to content]` |
| `read_03` | read | yes | `1413.68ms` | `3598` | `# Herman Melville - Moby-Dick` |
| `read_04` | read | yes | `19.25ms` | `365` | `[About](/info/en/about)[Preferences](/preferences)` |
| `read_05` | read | yes | `497.10ms` | `6000` | `{"query": "kazusa test", "number\_of\_results": 0, "results": \[{"url": "https://www.kazusa.or.jp/en-laboratories/en-gen` |
