# web_agent3 source availability bigbang plan

## Summary

- Goal: Replace web_agent3's legacy source roster with final explicit
  `web_read`, `web_search`, and config-gated `nhentai` sources.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang. No compatibility aliases, fallback source
  names, dual registration, or preservation of `generic`, `bilibili`, or
  `youtube` source names.
- Highest-risk areas: source/action mismatch, prompt roster drift, disabled
  sources appearing in router prompts, stale tests expecting legacy names,
  nHentai token loading bypassing `.env`, and accidental Reddit scope.
- Acceptance criteria: web_agent3 exposes only `web_read`, configured
  `web_search`, and configured `nhentai`; the router prompt contains only
  available sources; `generic`, `bilibili`, and `youtube` are absent; Reddit
  remains unsupported.

## Context

The web_agent3 ICD now defines a source-subagent creation guide. Current code
does not fully match it:

- `generic` combines URL read and SearXNG search, even though URL read is
  always available and SearXNG search depends on `SEARXNG_URL`.
- Missing `SEARXNG_URL` currently leaves generic search visible to the router
  and returns a late unavailable observation during execution.
- `nhentai` implements a good source-local provider, but it reads
  `NHENTAI_TOKEN` directly from process env and remains registered when the
  token is missing.
- `bilibili` and `youtube` are temporary generic-web fallback source modules.
  They are useful as dispatch proof points but are not final source-local
  providers.

The user selected a one-time finalizing change:

```text
web_read    -> direct URL read, available by default
web_search  -> SearXNG search, available only when SEARXNG_URL is configured
nhentai     -> nHentai metadata/search, available only when NHENTAI_TOKEN is configured
```

Reddit support is explicitly excluded because the Reddit interface is not
stable for this project yet. The previous Reddit plan has been removed from
the active plan registry.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing web_agent3 router contracts,
  prompt source roster rendering, source/action validation, or LLM-facing
  descriptions.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests that contain CJK
  prompt or description strings.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running any live LLM routing checks or writing LLM
  behavior review artifacts.

## Mandatory Rules

- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval and status `approved` or
  `in_progress`.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.
- Use parent-led native subagent execution. If native subagent capability is
  unavailable, stop before production implementation unless the user explicitly
  approves fallback execution.
- Use `venv\Scripts\python.exe` for Python commands.
- Do not read the real `.env` file during implementation or verification.
  Tests that need config use subprocess environment injection or monkeypatching.
- Implement this as a bigbang source contract update. Do not add alias modules,
  fallback mappers, compatibility source names, or dual registration.
- Keep source-specific execution inside source modules.
- Keep stable provider constants inside source modules.
- Put user-specific and deployment-specific configuration in `config.py`.
- Keep `NHENTAI_TOKEN` as the only nHentai user-specific config in this plan.
- Keep Reddit out of source modules, docs, tests, examples, and source
  registries.
- Keep web_agent3's public `WebAgent3().run(task, context, max_attempts)`
  contract unchanged.
- Keep router output shape as exactly `action`, `source`, and `query`.

## Must Do

- Verify the previous draft Reddit plan file and registry row remain absent.
- Replace `generic` source naming with final `web_read` and `web_search`
  sources.
- Delete `bilibili` and `youtube` from the available source roster.
- Add source availability support to subagent discovery.
- Add source/action pairing metadata or equivalent deterministic validation.
- Register `web_search` only when `SEARXNG_URL` is configured.
- Register `nhentai` only when `NHENTAI_TOKEN` is configured.
- Move only `NHENTAI_TOKEN` loading into `config.py`.
- Keep nHentai provider constants in `nhentai.py`.
- Update router prompt source text, initial state defaults, tests, ICD, and
  HOWTO for the final source roster.

## Deferred

- Do not add Reddit support.
- Do not create Bilibili or YouTube provider implementations.
- Do not preserve `generic`, `bilibili`, or `youtube` source names.
- Do not create compatibility aliases for removed sources.
- Do not move nHentai provider constants into `config.py`.
- Do not change L2d, resolver, RAG dispatcher, adapters, persistence,
  consolidation, scheduler, or database code.
- Do not add live external-service tests to the regular deterministic suite.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Source names | bigbang | Replace `generic` with `web_read` and `web_search`; remove `bilibili` and `youtube`. |
| Source discovery | bigbang | Register only enabled source modules. |
| Source/action validation | bigbang | Validate router decisions against enabled sources and supported actions. |
| nHentai config | bigbang | Load `NHENTAI_TOKEN` through `config.py`; use token presence for availability. |
| SearXNG config | bigbang | Use existing `SEARXNG_URL` as `web_search` availability. |
| Tests/docs | bigbang | Rewrite expectations to final names and availability states. |
| Reddit | no-op | Keep unsupported and unplanned. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- For bigbang areas, delete or rewrite legacy references instead of preserving
  them.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

Final source roster by configuration:

```text
always:
  web_read

when SEARXNG_URL is configured:
  web_search

when NHENTAI_TOKEN is configured:
  nhentai
```

Final source modules:

```text
subagent/web_read.py
subagent/web_search.py
subagent/nhentai.py
subagent/__init__.py
```

Final removed source modules:

```text
subagent/generic.py
subagent/bilibili.py
subagent/youtube.py
```

Router-visible source descriptions contain only enabled source names. `stop`
is graph-local: the executor returns a bounded stopped observation without
dispatching to a source subagent. Initial router state uses `web_read` for
stop placeholders. No final prompt, test, ICD section, or source registry
mentions `generic`, `bilibili`, `youtube`, or Reddit as available sources.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Source names | Use `web_read` and `web_search` | The names expose capability directly to the router model. |
| Cutover | Bigbang | The user requested final names with no backward compatibility. |
| Generic split | Create separate modules | Read and search have different configuration availability. |
| nHentai config | Move only `NHENTAI_TOKEN` to `config.py` | The token is user-specific; provider constants are stable implementation details. |
| Availability | Source discovery registers enabled modules only | Disabled tools stay out of prompt source text. |
| Bilibili/YouTube | Remove from available roster | Existing modules are temporary generic fallbacks, not final providers. |
| Reddit | Excluded | The interface is unstable and support is not wanted now. |

## Contracts And Data Shapes

Source modules expose:

```python
SOURCE: str
DESCRIPTION: str
SUPPORTED_ACTIONS: tuple[str, ...]
async def execute(decision: _RouterDecision) -> Any: ...
```

Configuration-dependent modules also expose:

```python
def is_enabled() -> bool: ...
```

Final action map:

```python
{
    "web_read": ("read",),
    "web_search": ("search",),
    "nhentai": ("read", "search"),
}
```

Router normalization uses enabled source names plus supported action metadata:

- `stop` decisions normalize to `action="stop"`, `source="web_read"`, and
  `query=""`.
- Non-stop decisions with an empty query normalize to `web_search` using
  `fallback_query` only when `web_search` is enabled; otherwise they normalize
  to `stop`.
- `read` with an enabled source that supports `read` preserves the source.
- `read` with a disabled, unknown, removed, or unsupported source normalizes
  to `web_read` only when the query is an HTTP(S) URL; otherwise it normalizes
  to `stop`.
- `search` with enabled `web_search` preserves `web_search`.
- `search` with enabled `nhentai` preserves `nhentai` only when the router
  explicitly emitted `source="nhentai"`.
- `search` with disabled `nhentai` normalizes to `stop`.
- `search` with an unknown, removed, or unsupported source normalizes to
  `web_search` only when `web_search` is enabled; otherwise it normalizes to
  `stop`.
- `nhentai` is never selected as a fallback source.

`SUPPORTED_ACTIONS` lists only source-executable actions. `stop` remains a
router action, but it is not a source action. `_tool_call_executor` handles
`stop` before source dispatch and records:

```python
{
    "status": "stopped",
    "source": "web_read",
    "action": "stop",
    "query": "",
    "message": "Router stopped without another web action.",
}
```

Normalization matrix:

| Raw action/source state | Required normalized decision |
|---|---|
| `stop` with any source/query | `stop`, `web_read`, empty query |
| invalid action with non-empty query | Treat as `search`, then apply search rows |
| `search`, explicit enabled `web_search` | Preserve `web_search` search |
| `search`, explicit enabled `nhentai` | Preserve `nhentai` search |
| `search`, explicit disabled `nhentai` | `stop`, `web_read`, empty query |
| `search`, unknown/removed/unsupported source and enabled `web_search` | `web_search` search |
| `search`, unknown/removed/unsupported source and disabled `web_search` | `stop`, `web_read`, empty query |
| `read`, explicit enabled `web_read` | Preserve `web_read` read |
| `read`, explicit enabled `nhentai` | Preserve `nhentai` read |
| `read`, explicit disabled `nhentai` | `stop`, `web_read`, empty query |
| `read`, unknown/removed/unsupported source with HTTP(S) URL query | `web_read` read |
| `read`, unknown/removed/unsupported source without HTTP(S) URL query | `stop`, `web_read`, empty query |
| `read` or `search` with empty query and enabled `web_search` | `web_search` search with `fallback_query` |
| `read` or `search` with empty query and disabled `web_search` | `stop`, `web_read`, empty query |

Removed names `generic`, `bilibili`, and `youtube` receive no special
compatibility handling. They are treated as unknown sources by the matrix.

`config.py` adds:

```python
NHENTAI_TOKEN = os.getenv("NHENTAI_TOKEN", "").strip()
NHENTAI_SOURCE_ENABLED = bool(NHENTAI_TOKEN)
```

The normal config load order remains:

```text
code default -> .env -> injected process environment
```

## LLM Call And Context Budget

The number of LLM calls is unchanged. This plan changes static router prompt
source text and deterministic validation only. It does not add a response-path
LLM call, background LLM call, retry loop, or repair call.

Affected response-path call:

| Stage | Route | Before | After | Budget |
|---|---|---|---|---|
| `_tool_call_generator` | `WEB_SEARCH_LLM` | One router call per retrieval attempt; static system prompt includes `generic`, `bilibili`, `youtube`, and configured `nhentai` source text. | One router call per retrieval attempt; static system prompt includes only enabled `web_read`, configured `web_search`, and configured `nhentai` source text. | At most `MAX_WEB_SEARCH_AGENT_RETRY` generator calls per `WebAgent3.run`; unchanged. |

Unchanged calls:

| Stage | Route | Change |
|---|---|---|
| `_tool_call_evaluator` | `WEB_SEARCH_LLM` | No prompt, payload, route, or retry-cap change. |
| `_tool_call_finalizer` | `WEB_SEARCH_LLM` | No prompt, payload, route, or token-budget change. |

Context inputs:

- System prompt changes only in the generated source roster under `# 来源原则`.
- Human payload remains `task`, projected `context`, `reference_time`,
  last three observations, and evaluator feedback.
- Prompt prefix remains static for a Python process because source availability
  is decided at import/discovery time from process configuration.
- Conservative static router prompt estimate stays under 12k characters with
  all optional sources enabled, below the default 50k-token planning cap.
- Prompt-render verification must confirm the final source roster appears and
  removed source names do not appear as source options.

Before:

```text
source text = generic, bilibili, youtube, nhentai
```

After, with all config present:

```text
source text = web_read, web_search, nhentai
```

After, with optional config absent:

```text
source text = web_read plus any configured optional sources
```

Prompt size decreases when optional sources are disabled. No context cap,
retry cap, model route, evaluator prompt, or finalizer prompt budget changes.

## Change Surface

Production-code subagent ownership is limited to `src/kazusa_ai_chatbot/config.py`
and `src/kazusa_ai_chatbot/rag/web_agent3/**`. Parent ownership covers tests,
documentation, registry rows, review evidence, and lifecycle updates.

### Verify Already Removed

- `development_plans/active/short_term/web_agent3_reddit_source_subagent_plan.md`
  - Verify the previous Reddit plan file remains absent.
- `development_plans/README.md`
  - Verify the previous Reddit plan row remains absent and this plan row is
    present.

### Delete

- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/generic.py`
  - Replaced by explicit final modules.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/bilibili.py`
  - Temporary fallback source removed from final roster.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/youtube.py`
  - Temporary fallback source removed from final roster.

### Create

- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/web_read.py`
  - Direct URL-read source.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/web_search.py`
  - SearXNG-backed search source gated by `SEARXNG_URL`.

### Modify

- `development_plans/README.md`
  - Remove Reddit plan row and add this plan row.
- `src/kazusa_ai_chatbot/config.py`
  - Add nHentai token config.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/__init__.py`
  - Add enabled-source discovery and supported-action registry.
- `src/kazusa_ai_chatbot/rag/web_agent3/subagent/nhentai.py`
  - Import token config, add availability, keep provider constants local.
- `src/kazusa_ai_chatbot/rag/web_agent3/contracts.py`
  - Update final source names and source/action normalization.
- `src/kazusa_ai_chatbot/rag/web_agent3/agent.py`
  - Update prompt source text, validation arguments, and initial stop source.
- `src/kazusa_ai_chatbot/rag/web_agent3/README.md`
  - Update source roster, create-subagent interface, and verification.
- `docs/HOWTO.md`
  - Document `SEARXNG_URL` and `NHENTAI_TOKEN` availability behavior.
- `tests/test_config.py`
  - Add nHentai token config tests.
- `tests/test_web_agent3.py`
  - Update source discovery, prompt source text, execution, and old-name
    absence tests.
- `tests/test_web_agent3_routing.py`
  - Update routing normalization and generator tests for final names.
- `tests/test_web_agent3_nhentai.py`
  - Update token handling and enabled-state coverage.

### Keep

- `WebAgent3().run(...)` public contract.
- `direct_searxng.py`, `url_reader.py`, and `searxng_tools.py` as lower-level
  execution helpers.
- nHentai provider constants and source-local compaction logic.

## Overdesign Guardrail

- Actual problem: web_agent3 exposes unavailable or placeholder sources to the
  router.
- Minimal change: rename/split generic web capabilities, gate optional sources
  at discovery, and remove temporary fallback sources from the final roster.
- Ownership boundaries: source modules own source execution; config owns
  user/deployment settings; discovery owns availability; router normalization
  owns source/action validation.
- Rejected complexity: compatibility aliases, fallback names, source-name
  translation, Reddit support, provider registries, browser automation, MCP
  dependency, database changes, and L2d/RAG changes.
- Evidence threshold: add a richer source metadata system only after another
  approved source needs metadata beyond `SOURCE`, `DESCRIPTION`,
  `SUPPORTED_ACTIONS`, and `is_enabled()`.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when all contracts
  in this plan are preserved.
- The responsible agent must not introduce compatibility layers, legacy source
  aliases, fallback mappers, extra source modules, or Reddit code.
- The responsible agent must treat changes outside the listed change surface
  as out of scope.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad refactors, or prompt rewrites.
- If the plan and code disagree, preserve this plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent adds failing tests for final discovery and config states.
   - Update `tests/test_config.py`, `tests/test_web_agent3.py`,
     `tests/test_web_agent3_routing.py`, and `tests/test_web_agent3_nhentai.py`.
   - Named tests to add or update:
     - `tests/test_config.py::TestDirectWebConfig::test_config_allows_empty_nhentai_token`
     - `tests/test_config.py::TestDirectWebConfig::test_config_reads_nhentai_token_from_environment`
     - `tests/test_web_agent3.py::test_web_agent3_source_discovery_registers_only_web_read_without_optional_config`
     - `tests/test_web_agent3.py::test_web_agent3_source_discovery_registers_web_search_when_searxng_configured`
     - `tests/test_web_agent3.py::test_web_agent3_source_discovery_registers_nhentai_when_token_configured`
     - `tests/test_web_agent3.py::test_web_agent3_source_discovery_registers_all_configured_sources`
     - `tests/test_web_agent3.py::test_web_agent3_router_prompt_lists_enabled_sources_only`
     - `tests/test_web_agent3.py::test_web_agent3_removed_source_modules_are_absent`
     - `tests/test_web_agent3_routing.py::test_web_agent3_router_normalizes_final_source_action_matrix`
     - `tests/test_web_agent3_routing.py::test_web_agent3_stop_bypasses_source_dispatch`
     - `tests/test_web_agent3_routing.py::test_web_agent3_generator_uses_final_enabled_sources`
     - `tests/test_web_agent3_nhentai.py::test_nhentai_is_disabled_without_token_config`
     - `tests/test_web_agent3_nhentai.py::test_nhentai_uses_token_from_config_headers`
     - `tests/test_web_agent3_nhentai.py::test_nhentai_provider_constants_remain_source_local`
   - Use subprocess environment injection or module re-import isolation for
     import-time config/discovery states. Do not read the real `.env`.
   - Expected failures: final source modules, enabled-source discovery,
     nHentai config constants, and the final normalization matrix do not exist
     yet.

2. Parent starts one production-code subagent.
   - Scope: `src/kazusa_ai_chatbot/config.py` and
     `src/kazusa_ai_chatbot/rag/web_agent3/**` production files only.
   - Parent-owned files outside the production subagent scope:
     `tests/**`, `docs/HOWTO.md`,
     `src/kazusa_ai_chatbot/rag/web_agent3/README.md`, and
     `development_plans/**`.
   - Production subagent edits production files only and reports changed files,
     commands run, blockers, and residual risks.

3. Production subagent implements config, discovery, and final source modules.
   - Add `NHENTAI_TOKEN` config.
   - Add enabled-source discovery and supported actions.
   - Create `web_read.py` and `web_search.py`.
   - Update `nhentai.py`.
   - Delete legacy source modules.

4. Production subagent updates router normalization and agent defaults.
   - Update final source names.
   - Enforce source/action pairing.
   - Make `stop` graph-local in `_tool_call_executor` before source dispatch.
   - Replace stop placeholders with `web_read`.

5. Parent updates docs and runs focused tests.
   - Update HOWTO and web_agent3 ICD.
   - Run focused config, discovery, routing, and source tests.

6. Parent runs all verification gates.
   - Run static greps, py_compile, focused tests, and regression tests.

7. Parent starts one independent code-review subagent.
   - Review plan alignment, source contract, prompt roster, tests, and docs.
   - Parent fixes approved-scope findings and reruns affected checks.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes focused failing tests before production
  implementation starts.
- Production-code subagent owns production code changes only under
  `src/kazusa_ai_chatbot/config.py` and
  `src/kazusa_ai_chatbot/rag/web_agent3/**`.
- Parent agent may update tests and docs while the production-code subagent
  edits production code.
- Independent code-review subagent reviews after planned verification passes
  and does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.
- Execution note: the user explicitly requested parent-only execution without
  subagents. The parent agent performed implementation and independent review.

## Progress Checklist

- [x] Stage 1 - focused tests established
  - Covers: implementation step 1.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_config.py::TestDirectWebConfig::test_config_allows_empty_nhentai_token tests\test_config.py::TestDirectWebConfig::test_config_reads_nhentai_token_from_environment tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_only_web_read_without_optional_config tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_web_search_when_searxng_configured tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_nhentai_when_token_configured tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_all_configured_sources tests\test_web_agent3.py::test_web_agent3_router_prompt_lists_enabled_sources_only tests\test_web_agent3.py::test_web_agent3_removed_source_modules_are_absent tests\test_web_agent3_routing.py::test_web_agent3_router_normalizes_final_source_action_matrix tests\test_web_agent3_routing.py::test_web_agent3_stop_bypasses_source_dispatch tests\test_web_agent3_routing.py::test_web_agent3_generator_uses_final_enabled_sources tests\test_web_agent3_nhentai.py::test_nhentai_is_disabled_without_token_config tests\test_web_agent3_nhentai.py::test_nhentai_uses_token_from_config_headers tests\test_web_agent3_nhentai.py::test_nhentai_provider_constants_remain_source_local -q`
  - Expected before implementation: selected tests fail for missing final
    source modules, config constants, availability gating, stop bypass, and
    normalization matrix behavior.
  - Evidence: focused contract tests were added and later passed as part of
    the 14-test focused run. Sign-off: `Codex/2026-06-29`.

- [x] Stage 2 - final source modules and discovery implemented
  - Covers: implementation steps 2-3.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_config.py::TestDirectWebConfig tests\test_web_agent3_nhentai.py -q`
  - Evidence: config, discovery, `web_read`, `web_search`, and `nhentai`
    source changes implemented; `tests\test_config.py::TestDirectWebConfig`
    and `tests\test_web_agent3_nhentai.py` passed. Sign-off:
    `Codex/2026-06-29`.

- [x] Stage 3 - router normalization and prompt roster updated
  - Covers: implementation step 4.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_only_web_read_without_optional_config tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_web_search_when_searxng_configured tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_nhentai_when_token_configured tests\test_web_agent3.py::test_web_agent3_source_discovery_registers_all_configured_sources tests\test_web_agent3.py::test_web_agent3_router_prompt_lists_enabled_sources_only tests\test_web_agent3.py::test_web_agent3_removed_source_modules_are_absent tests\test_web_agent3_routing.py::test_web_agent3_router_normalizes_final_source_action_matrix tests\test_web_agent3_routing.py::test_web_agent3_stop_bypasses_source_dispatch tests\test_web_agent3_routing.py::test_web_agent3_generator_uses_final_enabled_sources -q`
  - Evidence: router normalization, prompt roster, and stop bypass tests
    passed. Sign-off: `Codex/2026-06-29`.

- [x] Stage 4 - docs complete
  - Covers: implementation step 5.
  - Verify: static greps in `Verification`.
  - Evidence: HOWTO and web_agent3 ICD updated for final source roster and
    config-gated availability; doc greps passed. Sign-off:
    `Codex/2026-06-29`.

- [x] Stage 5 - full verification complete
  - Covers: implementation step 6.
  - Verify: every command in `Verification`.
  - Evidence: py_compile passed, targeted pytest suite reported 67 passed,
    static greps passed with expected no-match exit code 1, removed files were
    absent, and `git diff --check` exited 0 with line-ending warnings only.
    Sign-off: `Codex/2026-06-29`.

- [x] Stage 6 - independent code review complete
  - Covers: implementation step 7.
  - Verify: rerun affected checks after fixes.
  - Evidence: parent-only manual review found deterministic test coupling to
    ambient optional source registration and found router normalization used
    truthiness for optional source/action arguments; tests were patched to
    inject the provider registry explicitly, normalization now preserves
    explicit empty inputs, fallbacks require supported-action metadata, and
    affected plus full verification checks passed. Approval status: approved
    with no open blocker or major findings. Sign-off: `Codex/2026-06-29`.

## Verification

### Static Compile

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\rag\web_agent3\contracts.py src\kazusa_ai_chatbot\rag\web_agent3\agent.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\__init__.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\web_read.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\web_search.py src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`
  - Expected: command exits 0.

### Tests

- `venv\Scripts\python.exe -m pytest tests\test_config.py::TestDirectWebConfig tests\test_web_agent3_nhentai.py -q`
  - Expected: all tests pass.
- `venv\Scripts\python.exe -m pytest tests\test_web_agent3.py tests\test_web_agent3_routing.py -q`
  - Expected: all tests pass.
- `venv\Scripts\python.exe -m pytest tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py -q`
  - Expected: all tests pass.

### Static Greps

- `if ((Test-Path -LiteralPath 'src\kazusa_ai_chatbot\rag\web_agent3\subagent\generic.py') -or (Test-Path -LiteralPath 'src\kazusa_ai_chatbot\rag\web_agent3\subagent\bilibili.py') -or (Test-Path -LiteralPath 'src\kazusa_ai_chatbot\rag\web_agent3\subagent\youtube.py')) { exit 1 }`
  - Expected: command exits 0; removed source module files are absent.
- `rg -n "SOURCE = \"(generic|bilibili|youtube)\"|source=\"(generic|bilibili|youtube)\"|source: \"(generic|bilibili|youtube)\"|\"source\": \"(generic|bilibili|youtube)\"|subagent\.(generic|bilibili|youtube)" src\kazusa_ai_chatbot\rag\web_agent3 tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py docs\HOWTO.md`
  - Expected: no matches. This check targets source identifiers, source
    literals, and module imports only. `rg` exit code 1 is acceptable.
- ``rg -n '`generic`|`bilibili`|`youtube`|subagent/generic|subagent/bilibili|subagent/youtube|Bilibili|YouTube|generic web fallback|temporary generic web' src\kazusa_ai_chatbot\rag\web_agent3\README.md docs\HOWTO.md``
  - Expected: no matches. This check targets prose that would still describe
    removed source names or fallback adapters as available. Ordinary uses of
    the word `generic` outside these exact phrases are outside this check.
    `rg` exit code 1 is acceptable.
- `rg -ni "reddit" src\kazusa_ai_chatbot\rag\web_agent3 docs\HOWTO.md tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py`
  - Expected: no matches. `rg` exit code 1 is acceptable.
- `rg -n "os\.getenv\(\"NHENTAI_TOKEN\"" src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`
  - Expected: no matches. nHentai must import token config instead of reading
    process env directly.
- `rg -n "NHENTAI_TOKEN|NHENTAI_SOURCE_ENABLED" src\kazusa_ai_chatbot\config.py`
  - Expected: matches show token loading and enabled-state config in
    `config.py`.
- `rg -n "_NHENTAI_API_BASE_URL|_NHENTAI_PUBLIC_BASE_URL|_NHENTAI_USER_AGENT" src\kazusa_ai_chatbot\rag\web_agent3\subagent\nhentai.py`
  - Expected: matches show stable nHentai provider constants remain source
    local.
- `rg -n "web_agent3_reddit_source_subagent_plan|reddit_source_subagent" development_plans\README.md development_plans\active\short_term -g "!web_agent3_source_availability_bigbang_plan.md"`
  - Expected: no matches. This proves the removed Reddit plan identifier is
    absent from the registry and active plan set outside this plan. `rg` exit
    code 1 is acceptable.

### Diff Hygiene

- `git diff --check`
  - Expected: command exits 0.

## Independent Plan Review

Review gate performed by native subagent `019f12e7-2060-7743-b063-9d261d0cddad`
(`Jason`) before approval or execution. The review was read-only and returned
three blockers and four major findings.

Findings and remediation:

| Severity | Finding | Remediation in this draft |
|---|---|---|
| Blocker | `stop` was a router action but not a source action, and executor behavior after removing `generic` was undefined. | `Contracts And Data Shapes` now defines `stop` as graph-local, with executor bypass and a bounded stopped observation. |
| Blocker | Source/action fallback could accidentally route generic search intent to `nhentai`. | `Contracts And Data Shapes` now includes a normalization matrix and states `nhentai` is never selected as a fallback source. |
| Blocker | Production-code subagent scope included tests, docs, and registry files through broad `Change Surface` wording. | `Change Surface`, `Implementation Order`, and `Execution Model` now limit production subagent ownership to production paths under `src/`; parent owns tests, docs, registry, and evidence. |
| Major | Reddit plan deletion was listed as future required work even though the file and registry row were already absent. | `Must Do` and `Change Surface` now require verification that the previous Reddit plan file and registry row remain absent. |
| Major | Legacy-source grep used bare `generic`, causing noisy or vague verification. | `Verification` now separates removed module-file checks, exact source-literal/import checks, and case-insensitive Reddit checks. |
| Major | Focused test contract named files but not test functions or required config-state cases. | `Implementation Order` and `Progress Checklist` now name focused config, discovery, prompt roster, normalization, stop-bypass, removed-module, and nHentai tests. |
| Major | LLM budget omitted required prompt-change details. | `LLM Call And Context Budget` now names `_tool_call_generator`, route, call count, prompt-surface change, human payload, prefix-cache behavior, and conservative prompt-size estimate. |

Second review gate performed by native subagent
`019f12ef-731e-7c52-b4a9-025217abc62b` (`Dirac`) after remediation. The
second review found no blocker or major findings and approved the draft for
user approval. It surfaced one minor residual risk: stale prose-only removed
source mentions in docs could evade the code-literal grep. Remediation: the
`Verification` section now includes a targeted doc-prose grep for backticked
removed source names, removed module paths, Bilibili/YouTube prose, and generic
fallback phrases.

Status after remediation: all surfaced blocker, major, and minor findings have
corresponding plan edits. The user later approved execution and required
parent-only implementation without subagents.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Execution note: the user explicitly approved and required fallback execution
without subagents. The parent agent performed an independent manual review.

Review scope:

- Alignment with this plan's bigbang source roster and no-compatibility rule.
- Source availability behavior for `web_read`, `web_search`, and `nhentai`.
- Prompt source text and router normalization under missing optional config.
- Absence of `generic`, `bilibili`, `youtube`, and Reddit support paths.
- Test coverage for enabled/disabled registration states.
- Docs accuracy in HOWTO and the web_agent3 ICD.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `generic`, `bilibili`, and `youtube` source modules are removed.
- `web_read` is always registered.
- `web_search` is registered only when `SEARXNG_URL` is configured.
- `nhentai` is registered only when `NHENTAI_TOKEN` is configured.
- Router source text lists only enabled sources.
- Router normalization enforces final source/action pairings.
- `NHENTAI_TOKEN` is loaded through `config.py`.
- nHentai stable provider constants remain in `nhentai.py`.
- No Reddit source support, previous Reddit plan, Reddit docs/tests, or Reddit
  registry row remains active.
- Focused and regression tests pass.
- Independent code review is complete.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Router emits unavailable search source | Discovery and prompt text expose enabled sources only | Disabled `web_search` prompt test |
| Router emits mismatched source/action pair | Deterministic source/action validation | Routing tests |
| Legacy source name remains | Bigbang delete plus static grep | Legacy source grep |
| nHentai token bypasses `.env` loading | Token loaded through `config.py` | Config and grep tests |
| Reddit scope leaks back in | Verify previous plan and registry row remain absent | Reddit plan/source grep |

## Execution Evidence

- Status promoted to `completed` after user-approved parent-only execution on
  2026-06-29.
- Implemented final source roster:
  `web_read`, config-gated `web_search`, and config-gated `nhentai`.
- Removed source modules:
  `subagent/generic.py`, `subagent/bilibili.py`, and
  `subagent/youtube.py`.
- Moved only `NHENTAI_TOKEN` into `config.py`; nHentai provider constants
  remain source-local in `nhentai.py`.
- Verification:
  - `venv\Scripts\python.exe -m py_compile ...` exited 0.
  - `venv\Scripts\python.exe -m pytest tests\test_config.py::TestDirectWebConfig tests\test_web_agent3.py tests\test_web_agent3_routing.py tests\test_web_agent3_nhentai.py -q`
    reported 67 passed.
  - Removed-module path check exited 0.
  - Legacy source literal grep, doc-prose removed-source grep, Reddit grep,
    and nHentai direct getenv grep returned no matches.
  - Config token grep and nHentai provider constant grep returned expected
    matches.
  - Reddit plan identifier grep returned no matches outside this plan.
  - `git diff --check` exited 0 with Git line-ending warnings only.
- Independent manual code review:
  - Finding: executor tests for `web_search` could depend on ambient optional
    source registration from the current process config.
  - Fix: tests now inject the provider registry entries they exercise.
  - Finding: router normalization replaced explicitly empty source/action
    arguments with defaults and fallback routing did not require supported
    action metadata.
  - Fix: normalization now distinguishes `None` from empty inputs and requires
    action metadata for `web_read` and `web_search` fallback routing.
  - Rerun: affected routing and executor tests passed, then full targeted
    verification passed.
  - Residual risk: no live external-service checks were run; this matches the
    deterministic verification scope.
