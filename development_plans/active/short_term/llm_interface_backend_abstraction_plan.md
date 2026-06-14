# llm interface backend abstraction plan

## Summary

- Goal: Replace direct `get_llm()` / `ChatOpenAI` construction with a shared `LLInterface` backend compatibility layer while keeping stage/module-owned route and generation configuration explicit.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang
- Highest-risk areas: broad LLM call-site migration, provider request normalization, LM Studio reload retry preservation, JSON repair sync invocation, and route inventory/reporting continuity.
- Acceptance criteria: all production chat LLM construction goes through `LLInterface`; `get_llm()` has no shipped deprecation period and is removed before completion; no production import of `ChatOpenAI` or `get_llm()` remains outside the new provider adapter; deterministic and focused regression tests pass.
- External review state: conditionally approved after mandatory no-compatibility amendments. Lifecycle `Status` remains `draft` until explicit user approval.

## Context

Current chat LLM construction is centralized in `kazusa_ai_chatbot.utils.get_llm()`.
That function builds a `ChatOpenAI` instance, wraps it with
`MonitoredChatModel`, and returns an object exposing `.ainvoke()` and
`.invoke()`. Most modules keep their own module-level LLM instance, but the
factory itself is stateless. Reload coordination is shared through
`llm_reload_monitor.py` module state keyed by `(base_url, model)`.

The requested target architecture removes `get_llm()` as the public LLM
factory. The new boundary is not backward-compatible. Each module continues to
own the stage route choice and generation settings, including temperature,
top-p, top-k, token budget, backend hints, and thinking configuration read from
`config.py`. `LLInterface` owns backend compatibility only: backend detection,
provider client/session caching, request mapping, thinking payload mapping,
response normalization, retry/reload behavior, and backend cache invalidation.
The cognition-chain service contract must migrate to this explicit-config
interface instead of preserving an injected bound chat-model protocol.

This plan changes LLM infrastructure only. It must not change prompts,
semantic cognition behavior, RAG routing decisions, dialog policy, persistence
rules, adapter behavior, or JSON parsing semantics except where a call site is
rewired to the new invocation interface.

## Mandatory Skills

- `development-plan`: load before changing this plan, executing it, reviewing it, or marking lifecycle status.
- `local-llm-architecture`: load before changing LLM call boundaries, provider compatibility, cognition/RAG/dialog model invocation, or thinking behavior.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The execution model must use parent-led native subagent execution unless the user explicitly approves fallback execution.
- Do not preserve `get_llm()` as a compatibility shim. It may exist only as a short-lived migration marker inside an unchecked implementation stage and must be deleted before completion.
- Do not add bound or preconfigured chat clients that preserve the old call shape. Forbidden patterns include `LLInterface.bind(...)`, `ConfiguredLLMClient`, `get_llm_compat`, and adapters exposing only `.ainvoke(messages)` to avoid migrating call sites to `ainvoke(messages, *, config=...)`.
- Do not keep direct production `ChatOpenAI` construction outside the new provider adapter.
- Do not add prompt-level thinking instructions as a substitute for provider thinking configuration.
- Do not let `LLInterface` decide when a stage semantically needs thinking. Stage/module code supplies an `LLMCallConfig` whose thinking settings come from `config.py`.
- Do not add provider-specific request fields to cognition, RAG, dialog, consolidation, reflection, or background-work modules. Provider-specific mapping belongs in provider adapters.
- Keep LangGraph/tool `.ainvoke()` calls outside this migration. This plan applies only to chat LLM clients.
- Keep response-path LLM call counts unchanged. A call-site migration must not add new LLM calls, retries beyond the existing LM Studio unload behavior, or repair loops.
- Before editing files, check `git status --short` and preserve unrelated user changes.

## Must Do

- Complete a Stage 0 call-site and generation-parameter inventory before finalizing contracts. The inventory must list every production `get_llm()` call, every direct `ChatOpenAI` import/construction, every `MonitoredChatModel` or `monitored_chat_model` reference, every chat-model `.ainvoke()`/`.invoke()` call, every LangGraph/tool `.ainvoke()` that remains out of scope, and every generation kwarg currently passed to `get_llm()`, including `presence_penalty`.
- Create a new LLM interface package that owns backend compatibility and provider adapters.
- Define immutable module-owned config data shapes for route, model, generation settings, and thinking settings.
- Implement OpenAI-compatible provider support as the initial provider path.
- Preserve existing LM Studio unload retry coordination under the new session/provider layer.
- Rewire every production chat LLM call site that currently uses `get_llm()` to use `LLInterface` with a module-owned `LLMCallConfig`.
- Keep both async `.ainvoke()` and sync `.invoke()` because JSON repair uses synchronous invocation.
- Add a data-only route/backend diagnostics catalog that replaces the current route reporting dependency on scattered config constants and covers all chat routes, including background and artifact routes. It must report route name, backend, model, normalized base URL identity, model family, and thinking strategy. Diagnostics must not own prompts, route choice, generation policy, or stage behavior.
- Split session/client cache keys from diagnostic call-config fingerprints. Never log raw API keys.
- Migrate cognition-chain service contracts to an explicit-config invoker protocol instead of injecting bound chat-model objects.
- Remove `get_llm()` before completion. There is no shipped deprecation period, shim, alias, fallback import, or compatibility wrapper.
- Add focused tests for config fingerprinting, backend descriptor caching, thinking payload mapping, response normalization, reload retry behavior, and call-site migration.
- Add reload behavior tests proving owner retry, same-key waiters block, different model calls do not wait, and non-unload bad requests are not retried.
- Update README/HOWTO/runtime docs to describe the new LLM call architecture and config ownership.

## Deferred

- Do not implement native Anthropic support in this plan.
- Do not implement streaming, batch calls, tool calls, structured output helpers, or JSON parsing in `LLInterface`.
- Do not redesign prompt contracts, cognition stage schemas, RAG helper-agent semantics, dialog evaluator policy, or consolidation logic.
- Do not add a global stage registry that owns module temperatures or route choices.
- Do not make `LLInterface` a semantic router or model-quality chooser.
- Do not add compatibility aliases, fallback imports, or old `get_llm()` wrappers after cutover.
- Do not add `LLInterface.bind(...)`, `ConfiguredLLMClient`, or equivalent wrappers that preserve old `.ainvoke(messages)` call sites.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| LLM construction | bigbang | Replace `get_llm()` call sites with `LLInterface` and module-owned `LLMCallConfig`. |
| `get_llm()` | bigbang | Temporary working-tree only during cutover, then delete before completion. Do not retain as a shim or ship a deprecation period. |
| Provider implementation | bigbang | Move direct `ChatOpenAI` construction into the OpenAI-compatible provider adapter. |
| LM Studio unload retry | migration | Move the current behavior into the new session/provider layer and verify equivalent retry/wait behavior. |
| Module config ownership | bigbang | Modules continue to own route and generation config values read from `config.py`; `LLInterface` owns backend compatibility only. |
| Tests | bigbang | Rewrite tests that assert `get_llm()` behavior to assert the new interface and provider/session behavior. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `migration`, follow the exact migration phases and cleanup gates listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Production chat LLM calls use this shape:

```python
_llm_interface = LLInterface()

response = await _llm_interface.ainvoke(
    [system_prompt, human_message],
    config=_STAGE_LLM_CONFIG,
)
```

Each module owns its config object:

```python
_STAGE_LLM_CONFIG = LLMCallConfig(
    stage_name="cognition.layers",
    route_name="COGNITION_LLM",
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
    model=COGNITION_LLM_MODEL,
    backend_kind=COGNITION_LLM_BACKEND,
    model_family=COGNITION_LLM_MODEL_FAMILY,
    temperature=COGNITION_LLM_TEMPERATURE,
    top_p=COGNITION_LLM_TOP_P,
    top_k=COGNITION_LLM_TOP_K,
    max_tokens=COGNITION_LLM_MAX_TOKENS,
    presence_penalty=COGNITION_LLM_PRESENCE_PENALTY,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_THINKING_ENABLED,
        mode=COGNITION_LLM_THINKING_MODE,
        budget_tokens=COGNITION_LLM_THINKING_BUDGET_TOKENS,
        effort=COGNITION_LLM_THINKING_EFFORT,
    ),
)
```

`LLInterface` computes separate session/cache and diagnostic fingerprints from
the supplied config, resolves or creates a provider session, maps `BaseMessage`
values into the provider request shape, maps thinking config into
provider-specific payload fields, calls the backend, and returns a normalized
`LLMResponse`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Config ownership | Modules own `LLMCallConfig` instances. | Stage settings remain visible near the prompt/call site and are not hidden inside the compatibility engine. |
| Backend ownership | `LLInterface` owns backend/session/provider mapping. | Provider differences should not leak into cognition, RAG, dialog, or consolidation modules. |
| Thinking control | Thinking settings are read from `config.py` and passed via `LLMCallConfig`. | Deterministic config controls provider payloads; `LLInterface` does not make semantic decisions. |
| Dynamic thinking | Modules may choose between named config profiles. | This supports dynamic enabling without adding provider kwargs to call sites. |
| Call methods | Keep `.ainvoke()` and `.invoke()` on `LLInterface`. | Existing stage handlers use message-in/response-out semantics, and JSON repair needs sync invocation. |
| Initial provider | Implement OpenAI-compatible provider first. | Current runtime is OpenAI-compatible and can migrate without adding unrelated providers. |
| Anthropic | Design provider interface to allow a future native Anthropic adapter, but do not implement it. | Future support should be possible without expanding current scope. |
| `get_llm()` | Remove through big-bang migration with no shipped deprecation period. | User explicitly requested no backward compatibility. |
| Cognition-chain services | Use an explicit-config invoker protocol. | Migrates the contract instead of wrapping old injected model behavior. |
| Bound configured clients | Forbidden. | Prevents recreating the old `.ainvoke(messages)` call shape under a new name. |

## Contracts And Data Shapes

Public interface:

```python
class LLInterface:
    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        ...

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        ...

    def describe_backend(
        self,
        *,
        config: LLMCallConfig,
    ) -> BackendDescriptor:
        ...

    def invalidate_backend(
        self,
        *,
        route_name: str | None = None,
    ) -> None:
        ...

    async def aclose(self) -> None:
        ...
```

Config contract:

```python
@dataclass(frozen=True)
class LLMCallConfig:
    stage_name: str
    route_name: str
    base_url: str
    api_key: str
    model: str
    backend_kind: str
    model_family: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_tokens: int | None
    presence_penalty: float | None
    thinking: LLMThinkingConfig
```

`presence_penalty` is required because current dialog generation already passes
it through the old LLM construction path. Stage 0 must explicitly record whether
any current call site uses additional provider-neutral fields such as
`frequency_penalty` or `timeout`; add such fields only when existing behavior
requires them. Do not add arbitrary `**kwargs`.

The cognition-chain service contract must migrate from injected bound chat
models to an explicit-config protocol:

```python
class LLMInvoker(Protocol):
    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        ...

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse:
        ...


@dataclass(frozen=True)
class CognitionChainServices:
    llm: LLMInvoker
    cognition_config: LLMCallConfig
    boundary_core_config: LLMCallConfig
    action_selection_config: LLMCallConfig
    style_config: LLMCallConfig
    content_plan_config: LLMCallConfig
    preference_config: LLMCallConfig
    visual_config: LLMCallConfig
    parse_json: JsonParser
    logger: logging.Logger
```

Thinking contract:

```python
@dataclass(frozen=True)
class LLMThinkingConfig:
    enabled: bool
    mode: str
    budget_tokens: int | None
    effort: str | None
```

Backend descriptor:

```python
@dataclass(frozen=True)
class BackendDescriptor:
    route_name: str
    backend_kind: str
    model_family: str
    model: str
    normalized_base_url: str
    thinking_strategy: str
    confidence: str
    generation: int
```

Normalized response:

```python
@dataclass(frozen=True)
class LLMResponse:
    content: str
    backend: BackendDescriptor
    raw_response: object | None
    usage: Mapping[str, object]
```

Session/client cache key:

```python
(
    normalized_base_url,
    api_key_hash,
    model,
    backend_kind,
)
```

Diagnostic call-config fingerprint:

```python
(
    stage_name,
    route_name,
    normalized_base_url,
    model,
    backend_kind,
    model_family,
    temperature,
    top_p,
    top_k,
    max_tokens,
    presence_penalty,
    thinking.enabled,
    thinking.mode,
    thinking.budget_tokens,
    thinking.effort,
)
```

The session/client cache key is for provider client reuse. The diagnostic
fingerprint is for observability, route reporting, cache invalidation evidence,
and tests. Raw API keys must never appear in logs; use a stable hash where key
identity matters.

Forbidden compatibility shapes:

- Do not expose provider-native response objects as the main contract.
- Do not preserve `get_llm()` as a public API.
- Do not require modules to import provider adapters.
- Do not allow provider-specific kwargs at the LLM call site.
- Do not add `LLInterface.bind(...)`, `ConfiguredLLMClient`,
  `get_llm_compat`, or any adapter that keeps old `.ainvoke(messages)` usage.

## LLM Call And Context Budget

Before this plan, each affected stage makes one chat-model call through a
module-level object returned by `get_llm()`. After this plan, each affected
stage makes the same number of chat-model calls through `LLInterface`.

The migration must not add response-path calls. It changes client construction,
request payload mapping, thinking configuration, and response normalization
only. Prompt content, dynamic human payload construction, parser behavior, and
context budgets remain unchanged unless a call site must unwrap
`LLMResponse.content` instead of `response.content`.

Default context-window cap remains the current model-route behavior. This plan
does not raise prompt budgets or route caps. Token budget config moves from
factory defaults into module-owned `LLMCallConfig` values with a shared default
defined in config or LLM interface contracts.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/llm_interface/__init__.py`: public exports for the new LLM interface package.
- `src/kazusa_ai_chatbot/llm_interface/contracts.py`: dataclasses and protocols for `LLMCallConfig`, `LLMThinkingConfig`, `BackendDescriptor`, `LLMResponse`, and provider adapters.
- `src/kazusa_ai_chatbot/llm_interface/interface.py`: `LLInterface` public entrypoint and session lookup.
- `src/kazusa_ai_chatbot/llm_interface/session.py`: session cache keys, diagnostic fingerprints, backend descriptor cache, provider client/session cache, invalidation, and generation tracking.
- `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`: OpenAI-compatible provider adapter and `ChatOpenAI` construction.
- `src/kazusa_ai_chatbot/llm_interface/reload.py`: migrated LM Studio unload retry coordination from `llm_reload_monitor.py`.
- `src/kazusa_ai_chatbot/llm_interface/diagnostics.py`: data-only route/backend diagnostics for chat routes, including background and artifact routes.
- `tests/test_llm_interface_contracts.py`: focused contract and fingerprint tests.
- `tests/test_llm_interface_openai_provider.py`: provider request/response/thinking mapping tests using fakes.
- `tests/test_llm_interface_reload.py`: migrated unload retry behavior tests.
- `tests/test_llm_interface_migration.py`: static and import-boundary tests proving legacy call paths are gone.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add route-level backend, model-family, thinking, and generation-budget config values with explicit defaults where allowed by project config policy.
- `src/kazusa_ai_chatbot/llm_route_report.py`: render route table from module/config descriptors or the new interface contracts instead of assuming `get_llm()` construction.
- `src/kazusa_ai_chatbot/utils.py`: remove `get_llm()` and direct `ChatOpenAI` import after all call sites migrate; keep unrelated utility functions and JSON parsing behavior.
- `src/kazusa_ai_chatbot/llm_reload_monitor.py`: delete or replace with compatibility-free migrated module after references move to `llm_interface/reload.py`.
- Cognition-chain service contracts, service construction, and stage modules that currently receive injected chat-model objects: replace with `LLMInvoker` plus explicit per-stage `LLMCallConfig` fields.
- All production modules currently calling `get_llm()`: replace module-level factory calls with module-owned `LLMCallConfig` plus `LLInterface`.
- `README.md` and `docs/HOWTO.md`: document the new LLM route configuration and removal of `get_llm()`.
- `development_plans/README.md`: register this active draft plan.

### Delete

- `get_llm()` from `src/kazusa_ai_chatbot/utils.py`.
- Old tests whose only purpose is asserting `get_llm()` factory behavior, after equivalent new interface tests exist.
- Public use of `MonitoredChatModel` / `monitored_chat_model` if no longer referenced after reload behavior moves.

### Keep

- Existing prompt strings and prompt rendering contracts.
- Existing parser functions such as `parse_llm_json_output()` and `parse_json_with_llm()` semantics.
- Existing stage ownership: each prompt/call block remains local and inspectable.
- Existing route-specific environment variables for base URL, API key, and model.

## Overdesign Guardrail

- Actual problem: LLM invocation needs a backend-aware compatibility boundary that can support provider-specific thinking and future backend switching without leaking provider details into cognition, RAG, dialog, or consolidation modules.
- Minimal change: Replace `get_llm()` with `LLInterface` plus immutable module-owned config objects and one initial OpenAI-compatible provider adapter.
- Ownership boundaries: modules own route and generation config; `config.py` owns environment/config values; `LLInterface` owns backend compatibility and provider sessions; provider adapters own request/response mapping; deterministic code owns cache invalidation and retry mechanics; LLM prompts own semantic judgment only.
- Rejected complexity: no native Anthropic implementation, no streaming, no batch, no tool-call abstraction, no structured-output helper, no semantic routing, no global model-quality chooser, no `get_llm()` shim, and no prompt rewrites.
- Evidence threshold: add rejected complexity only after a concrete provider integration, call-site requirement, performance measurement, or approved follow-up plan requires it.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the target LLM interface and listed call sites as high-scrutiny changes.
- The responsible agent may remove legacy code only when greps and tests prove references are gone.
- If equivalent reload or route-report behavior already exists, migrate or move it into the new package instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If this plan and code disagree, preserve the plan's stated ownership split and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent completes Stage 0 inventory: production `get_llm()` calls, direct `ChatOpenAI` construction, monitored-model references, chat-model `.ainvoke()`/`.invoke()` calls, out-of-scope LangGraph/tool `.ainvoke()` calls, and all generation kwargs currently passed to `get_llm()`.
2. Parent records the non-chat `.ainvoke()` allowlist and confirms `LLMCallConfig` includes every existing provider-neutral generation field, including `presence_penalty`.
3. Parent adds focused tests for `LLMCallConfig`, `LLMThinkingConfig`, session cache key behavior, diagnostic fingerprint content, descriptor cache invalidation, and response normalization.
4. Parent runs those tests and records the expected missing-symbol failures.
5. Parent starts one production-code subagent for the new `llm_interface` package and reload migration only.
6. Production-code subagent creates the contracts, interface, session cache, OpenAI-compatible provider, diagnostics module, and reload module.
7. Parent reruns focused interface tests and records passing output.
8. Parent adds migration/static tests proving `get_llm()` call sites are replaced, forbidden compatibility shapes do not exist, and provider-specific imports stay inside `llm_interface`.
9. Parent or production-code subagent rewires module-level LLM instances and cognition-chain services to module-owned `LLMCallConfig` plus `LLInterface`.
10. Parent reruns focused migrated call-site tests after each group: utility JSON repair, cognition chain connector, dialog, RAG supervisor, RAG package workers, consolidation, reflection/background workers.
11. Parent deletes `get_llm()` and obsolete reload wrapper surfaces after static greps show no remaining production references.
12. Parent updates README/HOWTO and route reporting tests.
13. Parent runs full verification gates.
14. Parent starts one independent code-review subagent and records findings, fixes, reruns, and approval status.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 0 - current LLM surface inventoried
  - Covers: implementation steps 1-2.
  - Verify: recorded inventory includes `get_llm()`, direct `ChatOpenAI`, monitored-model references, chat-model invoke calls, non-chat `.ainvoke()` allowlist, and generation kwargs including `presence_penalty`.
  - Evidence: record inventory path or inline summary in `Execution Evidence`.
  - Sign-off: `<agent/date>` after inventory and evidence are recorded.

- [ ] Stage 1 - focused interface contract tests established
  - Covers: implementation steps 3-4.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py -q`.
  - Evidence: record expected missing-symbol failures or baseline output in `Execution Evidence`.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - `llm_interface` core implemented
  - Covers: implementation steps 5-7.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_reload.py -q`.
  - Evidence: record changed files and passing focused test output.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - production call sites migrated
  - Covers: implementation steps 8-10.
  - Verify: focused tests for JSON repair, cognition, dialog, RAG, consolidation, reflection, and background worker call sites pass.
  - Evidence: record static grep output and test commands run.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - strict `get_llm()` removal complete
  - Covers: implementation step 11.
  - Verify: `rg "get_llm\(|from kazusa_ai_chatbot\.utils import .*get_llm|ChatOpenAI|MonitoredChatModel|monitored_chat_model|ConfiguredLLMClient|LLInterface\.bind|get_llm_compat" src tests`.
  - Expected: no production matches outside the new provider/reload package and explicit migration tests.
  - Evidence: record grep output and allowed test-only matches.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - docs and route diagnostics updated
  - Covers: implementation step 12.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_route_report.py tests\test_llm_interface_migration.py -q`.
  - Evidence: record doc files changed and test output.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - full verification and independent code review complete
  - Covers: implementation steps 13-14.
  - Verify: all commands in `Verification` pass before review; rerun affected commands after review fixes.
  - Evidence: record review findings, fixes, reruns, residual risks, and approval status.
  - Sign-off: `<agent/date>` after verification, review, and evidence are recorded.

## Verification

### Static Greps

- `rg "get_llm\(" src tests`
  - Expected: no production matches; test matches allowed only when asserting the legacy symbol is absent.
- `rg "ChatOpenAI" src tests`
  - Expected: production match only in `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`; test matches allowed only in interface/provider tests.
- `rg "MonitoredChatModel|monitored_chat_model" src tests`
  - Expected: no production matches unless reload internals deliberately retain equivalent behavior under the new package name.
- `rg "AsyncChatModel|ConfiguredLLMClient|LLInterface\.bind|get_llm_compat" src tests`
  - Expected: no matches, unless `AsyncChatModel` has been renamed into an explicit-config protocol and no old `.ainvoke(messages)` shape remains.
- `rg "\.ainvoke\(" src/kazusa_ai_chatbot`
  - Expected: every chat LLM invocation uses `LLInterface.ainvoke(..., config=...)`. LangGraph/tool `.ainvoke()` calls must be listed in the Stage 0 allowlist and remain outside this migration.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py -q`
- `venv\Scripts\python -m pytest tests\test_llm_interface_openai_provider.py -q`
- `venv\Scripts\python -m pytest tests\test_llm_interface_reload.py -q`
- `venv\Scripts\python -m pytest tests\test_llm_interface_migration.py -q`
- `venv\Scripts\python -m pytest tests\test_utils.py -q`
- `venv\Scripts\python -m pytest tests\test_llm_route_report.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`

### Live LLM Smoke

Run live LLM tests one case at a time with output inspected:

- `venv\Scripts\python -m pytest -m live_llm tests/test_cognition_live_llm.py::test_live_msg_decontexualizer_returns_non_empty_output -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_call_rag_initializer_live_llm -q -s`

If live endpoints are unavailable, record the skip reason and do not claim live verification.

## Independent Plan Review

An external review report conditionally approved the architecture if mandatory
no-compatibility amendments are added before implementation. This draft now
incorporates those amendments:

- `get_llm()` is temporary working-tree code only and has no shipped
  deprecation period.
- Bound configured clients and compatibility wrappers are explicitly forbidden.
- Stage 0 inventories current call sites, direct providers, monitored wrappers,
  generation kwargs, and non-chat `.ainvoke()` allowlists.
- `presence_penalty` is included in `LLMCallConfig`.
- Session/client cache keys are split from diagnostic call-config fingerprints.
- Cognition-chain services migrate to explicit-config `LLMInvoker`.
- Reload equivalence and static grep verification are strengthened.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, provider-specific leaks, prompt changes, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused tests, static greps, live LLM smoke evidence, route report updates, and next-stage handoff notes.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Every production chat LLM client is invoked through `LLInterface`.
- `get_llm()` is removed from `utils.py`, has no production references, and has no shipped deprecation shim, alias, fallback import, or compatibility wrapper.
- Direct `ChatOpenAI` construction exists only in the OpenAI-compatible provider adapter.
- Module-owned `LLMCallConfig` objects carry route, generation, and thinking config from `config.py`.
- Cognition-chain service contracts use explicit per-stage `LLMCallConfig` fields and do not expose the old bound `.ainvoke(messages)` model shape.
- `LLInterface` owns backend descriptor caching, provider client/session caching, request mapping, thinking payload mapping, response normalization, and invalidation.
- Session/client cache keys and diagnostic fingerprints are separate, and raw API keys are never logged.
- Existing LM Studio unload retry behavior is preserved under the new session/provider layer.
- Data-only route diagnostics cover all chat routes, including background and artifact routes, without owning prompts, route choice, generation policy, or stage behavior.
- Prompt text, parser behavior, and response-path LLM call counts remain unchanged.
- Static greps, focused tests, deterministic regression tests, and available live LLM smoke tests pass or have recorded skips.
- Independent code review is complete with no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Broad call-site migration misses a module | Static grep for `get_llm(`, `ChatOpenAI`, and chat-model `.ainvoke()` patterns | Static grep gates and migration tests |
| Provider-specific fields leak into stage modules | Keep provider fields inside provider adapter; modules pass only `LLMCallConfig` | Provider-boundary tests and code review |
| Thinking config becomes semantic routing | Require modules to choose config profiles; `LLInterface` only maps config | Contract tests and code review |
| Reload retry behavior regresses | Move current tests to `test_llm_interface_reload.py` before changing implementation | Focused reload tests |
| JSON repair sync path breaks | Keep `.invoke()` on `LLInterface`; test `parse_json_with_llm()` behavior | `tests/test_utils.py` |
| Live LLM route diagnostics drift | Update route report tests and docs | `tests/test_llm_route_report.py` |
| Generation behavior regresses from dropped kwargs | Stage 0 inventory captures all current provider-neutral generation fields, starting with `presence_penalty` | Inventory evidence and contract tests |
| Cognition-chain old protocol survives under a new wrapper | Migrate service contracts to explicit `LLMInvoker` plus per-stage `LLMCallConfig` fields | Static greps, cognition-chain tests, and code review |

## Execution Evidence

- Static grep results:
- Focused test results:
- Regression test results:
- Live LLM smoke results or skips:
- Independent code review:
