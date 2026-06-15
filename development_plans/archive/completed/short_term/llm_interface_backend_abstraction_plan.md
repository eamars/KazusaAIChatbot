# llm interface backend abstraction plan

## Summary

- Goal: Replace direct `get_llm()` / `ChatOpenAI` construction with a shared `LLInterface` backend compatibility layer while keeping stage/module-owned route and generation configuration explicit.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang
- Highest-risk areas: broad LLM call-site migration, provider request normalization, LM Studio reload retry preservation, JSON repair sync invocation, cognition-chain service migration, and route inventory/reporting continuity including the `BACKGROUND_WORK_LLM` and legacy `BACKGROUND_ARTIFACT_LLM` routes.
- Acceptance criteria: all production chat LLM construction goes through `LLInterface`; `get_llm()` has no shipped deprecation period and is removed before completion; no production import of `ChatOpenAI` or `get_llm()` remains outside the new provider adapter; user-visible behavior has no intentional functional alteration; deterministic and focused regression tests pass.
- External review state: conditionally approved after mandatory no-compatibility amendments. User approved execution on 2026-06-15; lifecycle `Status` is `completed`.

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
top-p, top-k, `max_completion_tokens`, and boolean thinking configuration read
from `config.py`. `LLInterface` owns backend compatibility only: model-name
based backend/model-family detection, provider client/session caching, request
mapping, thinking payload mapping, response normalization, retry/reload
behavior, and backend cache invalidation.
The cognition-chain service contract must migrate to this explicit-config
interface instead of preserving an injected bound chat-model protocol.

This plan changes LLM infrastructure only. It must not change prompts,
semantic cognition behavior, RAG routing decisions, dialog policy, persistence
rules, adapter behavior, or JSON parsing semantics except where a call site is
rewired to the new invocation interface.

The functional baseline is user-visible equivalence. This is an interface
refactor, not a behavior refactor. The implementation must preserve the exact
model-facing message content for migrated calls, including system prompt text,
human message text, message order, and dynamic payload rendering. Any observed
live LLM output difference should first be investigated as a possible request
payload/config change before being accepted as ordinary model variance.

Current 2026-06-15 source inventory baseline:

- `rg "get_llm\(" src/kazusa_ai_chatbot tests` reports 72 textual matches,
  including the `get_llm()` definition.
- AST inventory reports 71 `get_llm()` call expressions: 60 under
  `src/kazusa_ai_chatbot` and 11 under `tests`.
- Production `get_llm()` call kwargs are limited to `temperature`, `top_p`,
  `model`, `base_url`, `api_key`, and one `presence_penalty` call in
  `nodes/dialog_agent.py`.
- Existing tests also cover legacy `max_tokens` and `max_completion_tokens`
  factory behavior. The new interface must use the unified
  `max_completion_tokens` name and must not add `max_tokens` to the new public
  config contract.
- `ChatOpenAI` construction is centralized in `utils.py`; `MonitoredChatModel`
  and `monitored_chat_model` live in `llm_reload_monitor.py`.
- `BACKGROUND_WORK_LLM` exists in `config.py` and background-work production
  modules but is not currently represented in `llm_route_report.py`,
  `README.md`, or `docs/HOWTO.md`. `BACKGROUND_ARTIFACT_LLM` remains the
  legacy fallback route.

## Mandatory Skills

- `development-plan`: load before changing this plan, executing it, reviewing it, or marking lifecycle status.
- `local-llm-architecture`: load before changing LLM call boundaries, provider compatibility, cognition/RAG/dialog model invocation, or thinking behavior.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the active execution owner must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- Execution staffing and delegation are intentionally outside this plan. The active execution owner must follow current user direction and harness policy while preserving this plan's contracts, verification gates, and evidence requirements.
- Do not preserve `get_llm()` as a compatibility shim. It may exist only as a short-lived migration marker inside an unchecked implementation stage and must be deleted before completion.
- Do not add bound or preconfigured chat clients that preserve the old call shape. Forbidden patterns include `LLInterface.bind(...)`, `ConfiguredLLMClient`, `get_llm_compat`, and adapters exposing only `.ainvoke(messages)` to avoid migrating call sites to `ainvoke(messages, *, config=...)`.
- Do not keep direct production `ChatOpenAI` construction outside the new provider adapter.
- Do not add prompt-level thinking instructions as a substitute for provider thinking configuration.
- Do not let `LLInterface` decide when a stage semantically needs thinking. Stage/module code supplies an `LLMCallConfig` whose boolean thinking setting comes from `config.py`.
- Initial thinking support is on/off only. All models default to thinking disabled. The only initially supported thinking payload mapping is Gemma 4; if thinking is enabled for any unsupported model, the provider adapter must omit thinking-specific payload fields and continue without changing prompts, messages, or call counts.
- Initial backend and model-family detection must be automatic and based on normalized model name only. Add a code comment `TODO: replace model-name detection with provider capability probing when backend probing is approved`; do not implement runtime backend probing in this plan.
- New public LLM config and provider payload code must use `max_completion_tokens`; do not introduce `max_tokens` as a new interface config field.
- `LLMCallConfig.api_key` must never appear in dataclass repr output, diagnostics, route reports, test failure text, or logs. Use an API-key hash only where key identity is needed.
- Do not add provider-specific request fields to cognition, RAG, dialog, consolidation, reflection, or background-work modules. Provider-specific mapping belongs in provider adapters.
- Keep LangGraph/tool `.ainvoke()` calls outside this migration. This plan applies only to chat LLM clients.
- Keep response-path LLM call counts unchanged. A call-site migration must not add new LLM calls, retries beyond the existing LM Studio unload behavior, or repair loops.
- Keep model-facing message content unchanged. Do not rewrite prompts, dynamic human payloads, message order, or prompt rendering while migrating the call interface.
- Before editing files, check `git status --short` and preserve unrelated user changes.

## Must Do

- Use the 2026-06-15 source inventory baseline in `Context` as the starting Stage 0 inventory. Before implementation, refresh it only if source files changed after this plan update. The refreshed inventory must still list every production `get_llm()` call, every direct `ChatOpenAI` import/construction, every `MonitoredChatModel` or `monitored_chat_model` reference, every chat-model `.ainvoke()`/`.invoke()` call, every LangGraph/tool `.ainvoke()` that remains out of scope, and every generation kwarg currently passed to `get_llm()`.
- Use the 2026-06-15 route inventory baseline in `Context` as the starting Stage 0 route inventory. Before implementation, refresh it only if route config, route diagnostics, README, or HOWTO changed after this plan update. The inventory must include every route constant in `config.py`, every row exposed by `llm_route_report.py`, every route used by production `get_llm()` calls, `BACKGROUND_WORK_LLM`, and the legacy `BACKGROUND_ARTIFACT_LLM` compatibility route.
- Create a new LLM interface package that owns backend compatibility and provider adapters.
- Define immutable module-owned config data shapes for route, model, generation settings, `max_completion_tokens`, and boolean thinking settings. Do not include module-owned backend-kind or model-family config fields.
- Implement OpenAI-compatible provider support as the initial provider path.
- Implement automatic model-name based backend/model-family detection in `LLInterface`, with a code TODO for future provider capability probing.
- Implement boolean thinking mapping for Gemma 4 only. Unsupported model families must ignore enabled thinking by omitting thinking-specific payload fields.
- Preserve existing LM Studio unload retry coordination under the new session/provider layer.
- Rewire every production chat LLM call site that currently uses `get_llm()` to use `LLInterface` with a module-owned `LLMCallConfig`.
- Keep both async `.ainvoke()` and sync `.invoke()` because JSON repair uses synchronous invocation.
- Add a data-only route/backend diagnostics catalog that replaces the current route reporting dependency on scattered config constants and covers all chat routes, including `BACKGROUND_WORK_LLM` and legacy `BACKGROUND_ARTIFACT_LLM`. It must report route name, backend, model, normalized base URL identity, model family, thinking strategy, and whether a route is required or fallback-backed. Diagnostics must not own prompts, route choice, generation policy, or stage behavior.
- Split per-interface session/client cache keys from diagnostic call-config fingerprints. Each `LLInterface()` instance owns its own backend descriptor cache and provider session/client cache. Never log raw API keys.
- Migrate cognition-chain service contracts to an explicit-config invoker protocol instead of injecting bound chat-model objects.
- Remove `get_llm()` before completion. There is no shipped deprecation period, shim, alias, fallback import, or compatibility wrapper.
- Add focused tests for config fingerprinting, API-key non-disclosure, per-interface backend descriptor caching, model-name detection, Gemma 4 thinking payload mapping, unsupported-model thinking ignore behavior, response normalization, reload retry behavior, and call-site migration.
- Add deterministic message/request equivalence tests for representative migrated call sites. These tests must compare the old and new rendered `BaseMessage` content, message order, route config, and provider request fields before any live LLM behavior is judged.
- Treat `tests/test_llm_interface_message_equivalence.py` as a temporary migration evidence file only. Before this plan is completed, remove that file after recording its evidence and move any durable guard assertions into `tests/test_llm_interface_migration.py`, `tests/test_llm_interface_contracts.py`, or provider tests.
- Add reload behavior tests proving owner retry, same-key waiters block, different model calls do not wait, and non-unload bad requests are not retried.
- Update README/HOWTO/runtime docs to describe the new LLM call architecture and config ownership.

## Deferred

- Do not implement native Anthropic support in this plan.
- Do not implement streaming, batch calls, tool calls, structured output helpers, or JSON parsing in `LLInterface`.
- Do not redesign prompt contracts, cognition stage schemas, RAG helper-agent semantics, dialog evaluator policy, or consolidation logic.
- Do not add a global stage registry that owns module temperatures or route choices.
- Do not make `LLInterface` a semantic router or model-quality chooser.
- Do not decommission the legacy `background_artifact` package or `BACKGROUND_ARTIFACT_LLM` compatibility route in this plan.
- Do not add compatibility aliases, fallback imports, or old `get_llm()` wrappers after cutover.
- Do not add `LLInterface.bind(...)`, `ConfiguredLLMClient`, or equivalent wrappers that preserve old `.ainvoke(messages)` call sites.
- Do not add backend-kind or model-family config constants to stage modules or `config.py`; backend and model family are detected by `LLInterface`.
- Do not implement thinking modes beyond boolean enabled/disabled or provider-specific thinking support beyond Gemma 4.
- Do not implement backend capability probing in this plan. The implementation must leave the explicit code TODO for a future approved probing change.

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
    temperature=COGNITION_LLM_TEMPERATURE,
    top_p=COGNITION_LLM_TOP_P,
    top_k=COGNITION_LLM_TOP_K,
    max_completion_tokens=COGNITION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=COGNITION_LLM_PRESENCE_PENALTY,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_THINKING_ENABLED,
    ),
)
```

`LLInterface` computes separate per-instance session/cache and diagnostic
fingerprints from the supplied config, detects backend/model-family from the
normalized model name, resolves or creates a provider session owned by that
`LLInterface` instance, maps `BaseMessage` values into the provider request
shape, maps boolean thinking config into Gemma 4 payload fields only when
supported, calls the backend, and returns a normalized `LLMResponse`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Config ownership | Modules own `LLMCallConfig` instances for route, model, generation settings, `max_completion_tokens`, and boolean thinking. | Stage settings remain visible near the prompt/call site and are not hidden inside the compatibility engine. |
| Backend ownership | `LLInterface` owns backend/session/provider mapping and model-name based detection. | Provider differences should not leak into cognition, RAG, dialog, or consolidation modules. |
| Backend detection | Initial detection is automatic from normalized model name only. Add a code TODO for future provider capability probing. | This satisfies dynamic backend identification without adding runtime probing or module-owned backend config in this plan. |
| Thinking control | Thinking is boolean on/off in config. All models default disabled; Gemma 4 is the only initial supported thinking mapping. | Deterministic config controls intent; provider adapters decide whether that intent can be mapped for the detected model family. |
| Dynamic thinking | Modules may dynamically pass boolean config profiles. Unsupported model families ignore enabled thinking by omitting thinking-specific payload fields. | This supports dynamic enabling without adding provider kwargs to call sites or breaking unsupported models. |
| Call methods | Keep `.ainvoke()` and `.invoke()` on `LLInterface`. | Existing stage handlers use message-in/response-out semantics, and JSON repair needs sync invocation. |
| Initial provider | Implement OpenAI-compatible provider first. | Current runtime is OpenAI-compatible and can migrate without adding unrelated providers. |
| Token budget naming | Use `max_completion_tokens` as the single public interface field. | `max_tokens` is not explicit enough for the new contract and must not be recreated as a public config field. |
| Cache scope | Each `LLInterface()` instance owns its backend descriptor and provider session/client cache. | Future control-console backend switching can invalidate one interface instance without relying on global singleton state. |
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
    api_key: str = field(repr=False)
    model: str
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_completion_tokens: int | None
    presence_penalty: float | None
    thinking: LLMThinkingConfig
```

`presence_penalty` is required because current dialog generation already passes
it through the old LLM construction path. The 2026-06-15 inventory found no
production `get_llm()` call kwargs beyond `temperature`, `top_p`, `model`,
`base_url`, `api_key`, and that one `presence_penalty`; if implementation-time
refresh finds an additional provider-neutral generation kwarg, add an explicit
field only when existing behavior requires it. Do not add arbitrary `**kwargs`.
The legacy factory tests cover both `max_tokens` and `max_completion_tokens`,
but the new public interface must expose only `max_completion_tokens`.

`LLMCallConfig.api_key` must use `field(repr=False)`, and contract tests must
prove the raw API key is absent from `repr(config)`, diagnostic fingerprints,
route diagnostics, and logs.

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
    enabled: bool = False
```

Initial thinking behavior:

- If `enabled` is false, no thinking-specific provider payload fields are sent.
- If `enabled` is true and detected `model_family` is `gemma4`, the
  OpenAI-compatible provider adapter sends the Gemma 4 thinking enablement
  payload required by the backend.
- If `enabled` is true for any other model family, the adapter sends no
  thinking-specific payload fields and continues the normal request.
- The implementation must add the code comment `TODO: replace model-name
  detection with provider capability probing when backend probing is approved`
  near the model-family detection function.

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

Initial descriptor values:

- `backend_kind`: `openai_compatible` for the initial provider path.
- `model_family`: `gemma4` when the normalized model name matches Gemma 4;
  `qwen`, `deepseek`, or `openai` when those family names are detected from the
  normalized model name; otherwise `unknown`.
- `thinking_strategy`: `disabled`, `gemma4_enabled`, or
  `ignored_unsupported_model`.
- `confidence`: `model_name_inferred` when a known family is detected from the
  model name; otherwise `unknown`.

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
    detected_backend_kind,
)
```

Diagnostic call-config fingerprint:

```python
(
    stage_name,
    route_name,
    normalized_base_url,
    model,
    detected_backend_kind,
    detected_model_family,
    temperature,
    top_p,
    top_k,
    max_completion_tokens,
    presence_penalty,
    thinking.enabled,
    effective_thinking_strategy,
)
```

The session/client cache key is for provider client reuse inside one
`LLInterface` instance. The diagnostic fingerprint is for observability, route
reporting, cache invalidation evidence, and tests. Raw API keys must never
appear in logs; use a stable hash where key identity matters.

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

Message payload equivalence is a hard deterministic requirement. For every
migrated chat LLM call, the new interface must receive the same ordered message
contents that the old `get_llm()`-returned model received for the same input.
Provider payload equivalence is also required for the OpenAI-compatible path:
model, base URL, temperature, top-p, `max_completion_tokens`,
`presence_penalty`, and message content must match existing behavior unless a
field is explicitly added only for disabled-by-default thinking support.

Default context-window cap remains the current model-route behavior. This plan
does not raise prompt budgets or route caps. Token budget config moves from
factory defaults into module-owned `LLMCallConfig` values with a shared default
defined in config or LLM interface contracts. New public config must use
`max_completion_tokens`, not `max_tokens`.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/llm_interface/__init__.py`: public exports for the new LLM interface package.
- `src/kazusa_ai_chatbot/llm_interface/contracts.py`: dataclasses and protocols for `LLMCallConfig`, `LLMThinkingConfig`, `BackendDescriptor`, `LLMResponse`, and provider adapters.
- `src/kazusa_ai_chatbot/llm_interface/interface.py`: `LLInterface` public entrypoint and session lookup.
- `src/kazusa_ai_chatbot/llm_interface/session.py`: per-interface session cache keys, diagnostic fingerprints, backend descriptor cache, provider client/session cache, invalidation, and generation tracking.
- `src/kazusa_ai_chatbot/llm_interface/detection.py`: model-name based backend and model-family detection, including the required provider-probing TODO comment.
- `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`: OpenAI-compatible provider adapter and `ChatOpenAI` construction.
- `src/kazusa_ai_chatbot/llm_interface/reload.py`: migrated LM Studio unload retry coordination from `llm_reload_monitor.py`.
- `src/kazusa_ai_chatbot/llm_interface/diagnostics.py`: data-only route/backend diagnostics for chat routes, including background and artifact routes.
- `tests/test_llm_interface_contracts.py`: focused contract and fingerprint tests.
- `tests/test_llm_interface_openai_provider.py`: provider request/response/thinking mapping tests using fakes.
- `tests/test_llm_interface_reload.py`: migrated unload retry behavior tests.
- `tests/test_llm_interface_migration.py`: static and import-boundary tests proving legacy call paths are gone.
- `tests/test_llm_interface_message_equivalence.py`: temporary migration-only evidence test, deleted before plan completion.

### Modify

- `src/kazusa_ai_chatbot/config.py`: add route-level boolean thinking and `max_completion_tokens` config values with explicit defaults where allowed by project config policy. Do not add backend-kind or model-family config values.
- `src/kazusa_ai_chatbot/llm_route_report.py`: render route table from module/config descriptors or the new interface contracts instead of assuming `get_llm()` construction. The route table must include `BACKGROUND_WORK_LLM` and the legacy `BACKGROUND_ARTIFACT_LLM` compatibility route.
- `src/kazusa_ai_chatbot/utils.py`: remove `get_llm()` and direct `ChatOpenAI` import after all call sites migrate; keep unrelated utility functions and JSON parsing behavior.
- `src/kazusa_ai_chatbot/llm_reload_monitor.py`: delete or replace with compatibility-free migrated module after references move to `llm_interface/reload.py`.
- Cognition-chain service contracts, service construction, and stage modules that currently receive injected chat-model objects: replace with `LLMInvoker` plus explicit per-stage `LLMCallConfig` fields.
- All production modules currently calling `get_llm()`: replace module-level factory calls with module-owned `LLMCallConfig` plus `LLInterface`.
- `README.md` and `docs/HOWTO.md`: document the new LLM route configuration and removal of `get_llm()`.
- `development_plans/README.md`: update only if lifecycle status or archive location changes; this active draft is already registered.

### Delete

- `get_llm()` from `src/kazusa_ai_chatbot/utils.py`.
- Old tests whose only purpose is asserting `get_llm()` factory behavior, after equivalent new interface tests exist.
- Public use of `MonitoredChatModel` / `monitored_chat_model` if no longer referenced after reload behavior moves.
- `tests/test_llm_interface_message_equivalence.py`, after its migration evidence is recorded and any durable assertions are folded into canonical interface tests.

### Keep

- Existing prompt strings and prompt rendering contracts.
- Existing model-facing message content and message order for every migrated
  chat LLM call.
- Existing parser functions such as `parse_llm_json_output()` and `parse_json_with_llm()` semantics.
- Existing stage ownership: each prompt/call block remains local and inspectable.
- Existing route-specific environment variables for base URL, API key, and model.
- Backend kind and model-family ownership inside `LLInterface`; stage modules and `config.py` do not own those values.
- Existing background route fallback behavior: `BACKGROUND_WORK_LLM` falls back through `BACKGROUND_ARTIFACT_LLM`, and `BACKGROUND_ARTIFACT_LLM` falls back through the cognition route.

## Overdesign Guardrail

- Actual problem: LLM invocation needs a backend-aware compatibility boundary that can support provider-specific thinking and future backend switching without leaking provider details into cognition, RAG, dialog, or consolidation modules.
- Minimal change: Replace `get_llm()` with `LLInterface` plus immutable module-owned config objects, model-name based backend detection, boolean thinking support for Gemma 4 only, and one initial OpenAI-compatible provider adapter.
- Ownership boundaries: modules own route and generation config; `config.py` owns environment/config values; `LLInterface` owns backend compatibility and provider sessions; provider adapters own request/response mapping; deterministic code owns cache invalidation and retry mechanics; LLM prompts own semantic judgment only.
- Rejected complexity: no native Anthropic implementation, no backend probing, no thinking modes beyond boolean on/off, no non-Gemma-4 thinking payload support, no streaming, no batch, no tool-call abstraction, no structured-output helper, no semantic routing, no global model-quality chooser, no `get_llm()` shim, and no prompt rewrites.
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

1. Refresh the 2026-06-15 inventory baseline only if relevant source, tests, docs, or route config changed after this plan update; otherwise record that the embedded baseline is current.
2. Add focused contract tests in `tests/test_llm_interface_contracts.py` for `LLMCallConfig`, `LLMThinkingConfig`, `max_completion_tokens`, API-key non-disclosure, per-interface cache separation, model-name detection, diagnostic fingerprint content, descriptor cache invalidation, response normalization, and deterministic message/request equivalence helper contracts.
3. Add focused provider tests in `tests/test_llm_interface_openai_provider.py` for OpenAI-compatible request mapping, Gemma 4 thinking payload enablement, unsupported-model thinking ignore behavior, and preservation of thinking-disabled payloads.
4. Run those focused tests and record the expected missing-symbol failures or baseline output.
5. Implement the new `llm_interface` package: contracts, interface, per-interface session cache, model-name detection with required probing TODO, OpenAI-compatible provider, diagnostics module, and reload module.
6. Rerun focused interface/provider/reload tests and record passing output.
7. Add migration/static tests proving `get_llm()` call sites are replaced, forbidden compatibility shapes do not exist, provider-specific imports stay inside `llm_interface`, no module-owned backend/model-family config exists, and representative migrated call sites preserve rendered message content.
8. Rewire module-level LLM instances and cognition-chain services to module-owned `LLMCallConfig` plus `LLInterface`.
9. Rerun focused migrated call-site tests after each group: utility JSON repair, cognition chain connector, dialog, RAG supervisor, RAG package workers, consolidation, reflection/background workers.
10. Run `tests/test_llm_interface_message_equivalence.py`, record its output as migration evidence, fold any durable assertions into canonical interface tests, and delete the temporary file.
11. Delete `get_llm()` and obsolete reload wrapper surfaces after static greps show no remaining production references.
12. Update README/HOWTO and route reporting tests.
13. Run full verification gates.
14. Run the independent code review gate and record findings, fixes, reruns, and approval status.

## Execution Model

- The active execution owner owns orchestration, test code, implementation coordination, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Execution staffing and delegation mechanics are intentionally not specified by this plan; follow current user direction and harness policy at execution time.
- The focused test contract must be established and run before production implementation starts.
- Production implementation must preserve the change surface, contracts, and no-functional-alteration baseline regardless of whether work is done by one agent or delegated.
- The independent code review gate must run after planned verification passes and before final sign-off; the review mechanism is intentionally not prescribed by this plan.

## Progress Checklist

- [x] Stage 0 - current LLM surface inventoried
  - Covers: implementation step 1.
  - Verify: record whether the embedded 2026-06-15 inventory baseline is still current. If source changed, refreshed inventory includes `get_llm()`, direct `ChatOpenAI`, monitored-model references, chat-model invoke calls, non-chat `.ainvoke()` allowlist, generation kwargs including `presence_penalty`, configured route constants, route-report rows, `BACKGROUND_WORK_LLM`, and legacy `BACKGROUND_ARTIFACT_LLM`.
  - Evidence: record the baseline-current statement or refreshed inventory path/summary in `Execution Evidence`.
  - Sign-off: `Codex / 2026-06-15` after inventory and evidence are recorded.

- [x] Stage 1 - focused interface contract tests established
  - Covers: implementation steps 2-4.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py -q`.
  - Evidence: record expected missing-symbol failures or baseline output in `Execution Evidence`, including `max_completion_tokens`, API-key non-disclosure, per-interface cache, model-name detection, Gemma 4 thinking, unsupported-model ignore behavior, and message/request equivalence helper coverage.
  - Sign-off: `Codex / 2026-06-15` after verification and evidence are recorded.

- [x] Stage 2 - `llm_interface` core implemented
  - Covers: implementation steps 5-6.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_reload.py -q`.
  - Evidence: record changed files and passing focused test output.
  - Sign-off: `Codex / 2026-06-15` after verification and evidence are recorded.

- [x] Stage 3 - production call sites migrated
  - Covers: implementation steps 7-10.
  - Verify: focused tests for JSON repair, cognition, dialog, RAG, consolidation, reflection, and background worker call sites pass, including representative rendered-message equivalence assertions; `tests/test_llm_interface_message_equivalence.py` is run once as temporary migration evidence before deletion.
  - Evidence: record static grep output, message/request equivalence output, cleanup of the temporary equivalence file, and test commands run.
  - Sign-off: `Codex / 2026-06-15` after verification and evidence are recorded.

- [x] Stage 4 - strict `get_llm()` removal complete
  - Covers: implementation step 11.
  - Verify: `rg "get_llm\(|from kazusa_ai_chatbot\.utils import .*get_llm|ChatOpenAI|MonitoredChatModel|monitored_chat_model|ConfiguredLLMClient|LLInterface\.bind|get_llm_compat" src tests`.
  - Expected: no production matches outside the new provider/reload package and explicit migration tests.
  - Evidence: record grep output and allowed test-only matches.
  - Sign-off: `Codex / 2026-06-15` after verification and evidence are recorded.

- [x] Stage 5 - docs and route diagnostics updated
  - Covers: implementation step 12.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_route_report.py tests\test_llm_interface_migration.py -q`.
  - Evidence: record doc files changed and test output.
  - Sign-off: `Codex / 2026-06-15` after verification and evidence are recorded.

- [x] Stage 6 - full verification and independent code review complete
  - Covers: implementation steps 13-14.
  - Verify: all commands in `Verification` pass before review; rerun affected commands after review fixes.
  - Evidence: record review findings, fixes, reruns, residual risks, and approval status.
  - Sign-off: `Codex / 2026-06-15` after verification, review, and evidence are recorded.

## Verification

### Static Greps And Cleanup Checks

- `rg "get_llm\(" src tests`
  - Expected: no production matches; test matches allowed only when asserting the legacy symbol is absent.
- `rg "ChatOpenAI" src tests`
  - Expected: production match only in `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`; test matches allowed only in interface/provider tests.
- `rg "MonitoredChatModel|monitored_chat_model" src tests`
  - Expected: no production matches unless reload internals deliberately retain equivalent behavior under the new package name.
- `rg "AsyncChatModel|ConfiguredLLMClient|LLInterface\.bind|get_llm_compat" src tests`
  - Expected: no matches, unless `AsyncChatModel` has been renamed into an explicit-config protocol and no old `.ainvoke(messages)` shape remains.
- `rg "LLM_.*(BACKEND|MODEL_FAMILY)|BACKEND_KIND|MODEL_FAMILY" src/kazusa_ai_chatbot/config.py`
  - Expected: no matches; backend kind and model family are detected by `LLInterface`, not configured by modules.
- `rg "max_tokens" src/kazusa_ai_chatbot/llm_interface tests/test_llm_interface_*.py`
  - Expected: no public-interface config field named `max_tokens`; matches allowed only in tests asserting that no public `max_tokens` config exists.
- `rg "\.ainvoke\(" src/kazusa_ai_chatbot`
  - Expected: every chat LLM invocation uses `LLInterface.ainvoke(..., config=...)`. LangGraph/tool `.ainvoke()` calls must be listed in the Stage 0 allowlist and remain outside this migration.
- `rg "BACKGROUND_WORK_LLM|BACKGROUND_ARTIFACT_LLM" src/kazusa_ai_chatbot README.md docs/HOWTO.md`
  - Expected: `BACKGROUND_WORK_LLM` is present in config, production background-work call sites, route diagnostics, and docs. `BACKGROUND_ARTIFACT_LLM` remains documented as a legacy compatibility/fallback route only.
- `if (Test-Path -LiteralPath 'tests\test_llm_interface_message_equivalence.py') { throw 'temporary message equivalence test file must be removed before completion' }`
  - Expected: no file exists at `tests/test_llm_interface_message_equivalence.py` after its migration evidence has been recorded.

### Temporary Migration Evidence

- `venv\Scripts\python -m pytest tests\test_llm_interface_message_equivalence.py -q`
  - Run before cleanup to prove representative message/request equivalence, record the output in `Execution Evidence`, then delete this temporary test file before final verification.

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
- `venv\Scripts\python -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_current_time_to_runtime_live_context -q -s`

Live LLM pass criteria follow `.agents/skills/test-style-and-execution`: harness
gates prove only that the test ran, contract gates check schema/safety/literal
preservation, and behavioral criteria require human or agent judgment from the
trace. Do not use exact prose matching as the primary pass criterion.

For this refactor, live LLM tests are functional smoke evidence after
deterministic message/request equivalence has passed. If a live output appears
functionally different, first compare the captured model-facing messages,
message order, model config, and provider payload against the pre-refactor
baseline. Treat a changed prompt/message payload as a migration defect, not as
ordinary LLM variance.

If live endpoints are unavailable, record the skip reason and do not claim live verification.

## Independent Plan Review

An external review report conditionally approved the architecture if mandatory
no-compatibility amendments are added before implementation. This draft now
incorporates those amendments:

- `get_llm()` is temporary working-tree code only and has no shipped
  deprecation period.
- Bound configured clients and compatibility wrappers are explicitly forbidden.
- Stage 0 inventories current call sites, direct providers, monitored wrappers,
  generation kwargs, configured route constants, route-report rows, and
  non-chat `.ainvoke()` allowlists.
- `presence_penalty` is included in `LLMCallConfig`.
- The source inventory baseline was updated on 2026-06-15: 71 total
  `get_llm()` call expressions, 60 under `src/kazusa_ai_chatbot`, 11 under
  `tests`, and production kwargs limited to the fields named in `Context`.
- The public token budget field is `max_completion_tokens`; `max_tokens` is not
  part of the new public config contract.
- Backend/model-family detection is automatic from normalized model name.
- Thinking is boolean on/off, defaults disabled, and initially maps only for
  Gemma 4; unsupported model families ignore enabled thinking by omitting
  thinking payload fields.
- Session/client cache keys are per-interface and split from diagnostic
  call-config fingerprints.
- Raw API keys are excluded from repr, diagnostics, route reports, test failure
  text, and logs.
- Cognition-chain services migrate to explicit-config `LLMInvoker`.
- Reload equivalence and static grep verification are strengthened.
- The 2026-06-15 source review identified `BACKGROUND_WORK_LLM` as a new
  first-class route and `BACKGROUND_ARTIFACT_LLM` as a legacy compatibility
  route; this plan now requires both to be represented in diagnostics and docs.
- The user-visible functional baseline is no intentional behavior change. This
  plan now requires deterministic message/request equivalence before relying on
  real LLM smoke-test judgment.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Use the current user-approved review mechanism available at execution time.
This plan intentionally does not prescribe the reviewer type or delegation
mechanism.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, provider-specific leaks, prompt changes, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused tests, static greps, live LLM smoke evidence, route report updates, next-stage handoff notes, and proof that `tests/test_llm_interface_message_equivalence.py` was removed before completion.

The active execution owner fixes concrete findings directly only when the fix
is inside the approved change surface or this review gate explicitly allows
review-only fixture/documentation corrections. If a fix would cross the
approved boundary or alter the contract, stop and update the plan or request
approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Every production chat LLM client is invoked through `LLInterface`.
- `get_llm()` is removed from `utils.py`, has no production references, and has no shipped deprecation shim, alias, fallback import, or compatibility wrapper.
- Direct `ChatOpenAI` construction exists only in the OpenAI-compatible provider adapter.
- Module-owned `LLMCallConfig` objects carry route, generation, and thinking config from `config.py`.
- `LLMCallConfig` uses `max_completion_tokens` as the only public token-budget field and does not expose `max_tokens`.
- Backend kind and model family are detected by `LLInterface` from normalized model name, not owned by `config.py` or stage modules.
- Initial thinking behavior is boolean only: disabled by default, Gemma 4 payload mapping when enabled and supported, and no thinking payload fields for unsupported model families.
- Cognition-chain service contracts use explicit per-stage `LLMCallConfig` fields and do not expose the old bound `.ainvoke(messages)` model shape.
- `LLInterface` owns per-interface backend descriptor caching, provider client/session caching, request mapping, thinking payload mapping, response normalization, and invalidation.
- Session/client cache keys and diagnostic fingerprints are separate, and raw API keys are never logged.
- Existing LM Studio unload retry behavior is preserved under the new session/provider layer.
- Data-only route diagnostics cover all chat routes, including `BACKGROUND_WORK_LLM` and legacy `BACKGROUND_ARTIFACT_LLM`, without owning prompts, route choice, generation policy, or stage behavior.
- Existing background route fallback behavior is preserved: `BACKGROUND_WORK_LLM` falls back through `BACKGROUND_ARTIFACT_LLM`, and `BACKGROUND_ARTIFACT_LLM` falls back through the cognition route.
- Prompt text, model-facing message content, message order, parser behavior, and response-path LLM call counts remain unchanged.
- OpenAI-compatible provider payloads preserve existing model, base URL, temperature, top-p, `max_completion_tokens`, `presence_penalty`, and message content for thinking-disabled calls.
- Raw API keys are absent from `LLMCallConfig` repr output, diagnostics, route reports, test failure text, and logs.
- Real LLM smoke results are judged against the updated real-LLM pass criteria guidance, with functional differences investigated first as possible message/config/payload drift.
- Temporary migration evidence from `tests/test_llm_interface_message_equivalence.py` is recorded, any durable assertions are moved into canonical interface tests, and `tests/test_llm_interface_message_equivalence.py` is absent from the completed codebase.
- Static greps, focused tests, deterministic regression tests, and available live LLM smoke tests pass or have recorded skips.
- Independent code review is complete with no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Broad call-site migration misses a module | Static grep for `get_llm(`, `ChatOpenAI`, and chat-model `.ainvoke()` patterns | Static grep gates and migration tests |
| Provider-specific fields leak into stage modules | Keep provider fields inside provider adapter; modules pass only `LLMCallConfig` | Provider-boundary tests and code review |
| Thinking config becomes semantic routing | Require modules to choose config profiles; `LLInterface` only maps config | Contract tests and code review |
| Thinking is accidentally sent for unsupported models | Support Gemma 4 thinking payloads only and omit thinking payload fields for unsupported model families | Provider tests for unsupported-model ignore behavior |
| Backend model-name detection misclassifies a family | Keep initial detection deterministic and diagnostic-only except for Gemma 4 thinking support; add explicit code TODO for future probing | Model-name detection tests and descriptor diagnostics |
| Raw API key leaks through dataclass repr, diagnostics, or logs | Use `field(repr=False)` and API-key hashes only | Contract tests for repr, fingerprints, diagnostics, and captured logs |
| Global cache prevents future dynamic backend switching | Scope backend descriptor and provider session/client caches to each `LLInterface()` instance | Per-interface cache separation tests |
| Reload retry behavior regresses | Move current tests to `test_llm_interface_reload.py` before changing implementation | Focused reload tests |
| JSON repair sync path breaks | Keep `.invoke()` on `LLInterface`; test `parse_json_with_llm()` behavior | `tests/test_utils.py` |
| Live LLM route diagnostics drift | Update route report tests and docs | `tests/test_llm_route_report.py` |
| Generation behavior regresses from dropped kwargs | Stage 0 inventory captures all current provider-neutral generation fields, starting with `presence_penalty` | Inventory evidence and contract tests |
| Cognition-chain old protocol survives under a new wrapper | Migrate service contracts to explicit `LLMInvoker` plus per-stage `LLMCallConfig` fields | Static greps, cognition-chain tests, and code review |
| Background-work route omitted from diagnostics | Stage 0 compares config constants, route-report rows, and production call sites, then requires explicit `BACKGROUND_WORK_LLM` coverage | Static grep and route-report tests |
| Interface migration changes prompt/message payloads | Require deterministic message/request equivalence tests before live LLM smoke; keep durable assertions in canonical interface tests after temporary file deletion | Temporary message-equivalence run output and representative call-site tests |
| Temporary message-equivalence test survives as permanent test surface | Run it once for migration evidence, fold durable assertions into canonical interface tests, then delete it before final verification | Static `Test-Path` cleanup gate and execution evidence |
| Live LLM output drift is misclassified as acceptable variance | Use real LLM pass criteria guidance and inspect captured messages/config first | Live trace review plus deterministic equivalence artifacts |

## Execution Evidence

- Static grep results:
  - `rg "get_llm\(|from kazusa_ai_chatbot\.utils import .*get_llm|ChatOpenAI|MonitoredChatModel|monitored_chat_model|ConfiguredLLMClient|LLInterface\.bind|get_llm_compat" src tests -n`: production matches only in `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py` for `ChatOpenAI`; legacy markers only in migration tests.
  - `rg "LLM_.*(BACKEND|MODEL_FAMILY)|BACKEND_KIND|MODEL_FAMILY" src\kazusa_ai_chatbot\config.py -n`: no matches.
  - `rg "max_tokens" src\kazusa_ai_chatbot\llm_interface tests --glob "test_llm_interface_*.py" -n`: matches only tests asserting the field is absent.
  - `rg "TODO: replace model-name detection with provider capability probing" src\kazusa_ai_chatbot\llm_interface -n`: required TODO present in `detection.py`.
  - `rg "BACKGROUND_WORK_LLM|BACKGROUND_ARTIFACT_LLM" src\kazusa_ai_chatbot README.md docs\HOWTO.md -n`: routes present in config, production call sites, route diagnostics, README, and HOWTO.
  - `rg "\.ainvoke\(|\.invoke\(" src\kazusa_ai_chatbot -n`: chat LLM calls use `LLInterface` with `config=...`; remaining no-config calls are LangGraph/tool calls or provider/reload internals.
- Focused test results:
  - `venv\Scripts\python -m pytest tests\test_llm_route_report.py tests\test_llm_interface_contracts.py tests\test_llm_interface_openai_provider.py tests\test_llm_interface_reload.py tests\test_llm_interface_migration.py tests\test_utils.py -q`: 44 passed, 3 deselected.
  - `venv\Scripts\python -m compileall -q src tests\test_adapter_readable_mentions_live_llm.py tests\test_conversation_progress_recorder_live_llm.py tests\test_decontexualizer_live_llm.py tests\test_l2d_action_selection_live_llm.py tests\test_l2d_quiet_monologue_live_llm.py tests\test_l2d_unknown_context_resolver_live_llm.py tests\test_persona_supervisor2_decontext_scope_users_live_llm.py tests\test_cognition_live_llm.py tests\test_utils.py tests\test_llm_interface_contracts.py tests\test_llm_interface_migration.py`: passed.
  - `venv\Scripts\python -m pytest -m live_llm tests\test_adapter_readable_mentions_live_llm.py tests\test_cognition_live_llm.py tests\test_l2d_action_selection_live_llm.py --collect-only -q`: 7 live tests collected.
- Temporary message/request equivalence evidence and file-removal proof:
  - Temporary `tests/test_llm_interface_message_equivalence.py` was run before cleanup: 2 passed.
  - Durable representative equivalence moved into `tests/test_llm_interface_migration.py::test_json_repair_call_preserves_messages_and_route_config`.
  - `if (Test-Path -LiteralPath 'tests\test_llm_interface_message_equivalence.py') { throw 'temporary message equivalence test file must be removed before completion' } else { 'temporary file absent' }`: `temporary file absent`.
- Regression test results:
  - `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q --tb=short --maxfail=50`: 2214 passed, 303 deselected.
- Live LLM smoke results or skips:
  - `venv\Scripts\python -m pytest -m live_llm tests\test_cognition_live_llm.py::test_live_msg_decontexualizer_returns_non_empty_output -q -s --tb=short`: 1 passed. Output resolved `他` to the classmate from prior context and returned a non-empty decontextualized input.
  - `venv\Scripts\python -m pytest -m live_llm tests\test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_current_time_to_runtime_live_context -q -s --tb=short`: 1 passed. Output selected `Live-context: answer active character current local time`.
  - Review-fix smoke: `venv\Scripts\python -m pytest -m live_llm tests\test_cognition_live_llm.py::test_live_cognition_stack_exercises_each_stage_llm -q -s --tb=short`: 1 passed. L1-L4 stage outputs were populated, natural-English preference was accepted, and content plan was valid.
  - Review-fix smoke: `venv\Scripts\python -m pytest -m live_llm tests\test_adapter_readable_mentions_live_llm.py::test_live_adapter_readable_mentions_drive_person_context -q -s --tb=short`: 1 passed. Readable `@蚝爹油` mention survived decontextualization and routed to a `Person-context` slot.
- Independent code review:
  - Production-code subagent: one `gpt-5.5 high` worker pass, agent `019ec8cf-5992-7081-bd6e-a3655ad3e5b9`, closed after production changes.
  - Code-review subagent: one `gpt-5.5 xhigh` review pass, agent `019ec8fc-608b-7cc3-beb4-88b9bfa0eade`, closed after reporting.
  - Review findings fixed: route-scoped invalidation now evicts shared provider sessions and has a regression test; durable message/config equivalence was added to canonical migration tests; stale live wrapper delegation now forwards `config`; stale cognition-stage live bindings now use `LLMStageBinding`; stale adapter-readable RAG initializer patch target was corrected; live-smoke plan command was updated.
  - Residual risk: live LLM output remains model-variant by nature, but deterministic message/config equivalence and provider payload mapping passed before live smoke.
