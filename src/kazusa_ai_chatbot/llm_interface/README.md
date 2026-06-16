# LLM Interface Control Document

## Document Control

- ICD id: `LLM-ICD-001`
- Owning package: `kazusa_ai_chatbot.llm_interface`
- Interface boundary: LLM-using runtime modules -> backend-compatible chat LLM
  invocation
- Runtime consumers: relevance, decontextualization, RAG, cognition,
  dialog, consolidation, reflection, background work, global character growth,
  and JSON repair stages
- Backend owner: `llm_interface` internals, including provider adapters,
  backend/model-family detection, provider session caches, reload retry, and
  response normalization

This document defines the chat LLM invocation boundary for Kazusa. It is the
source of truth for how runtime modules call model backends without owning
provider-specific request details.

For route-specific environment variables and operator setup, use the
operational [HOWTO](../../../docs/HOWTO.md). This ICD owns the runtime
interface contract.

## Purpose

`llm_interface` is the compatibility layer between Kazusa's semantic stages
and OpenAI-compatible chat model backends. It replaces direct model factory
usage with an explicit per-call interface:

```python
response = await _stage_llm.ainvoke(
    [system_prompt, human_message],
    config=_stage_llm_config,
)
```

The package keeps model-facing prompt construction in the calling module while
centralizing backend mechanics: model-name detection, provider sessions,
thinking payload mapping, response normalization, and LM Studio unload retry.

The interface is deliberately not a semantic router. It does not choose which
stage should use which route, prompt, model, temperature, or thinking setting.

## Scope

This ICD covers:

- Public contracts exported by `kazusa_ai_chatbot.llm_interface`.
- `LLInterface.ainvoke(...)`, `LLInterface.invoke(...)`,
  `describe_backend(...)`, `invalidate_backend(...)`, and `aclose()`.
- `LLMCallConfig`, `LLMThinkingConfig`, `BackendDescriptor`, `LLMResponse`,
  and `LLMInvoker`.
- Provider adapter ownership and request mapping.
- Backend and model-family detection.
- Per-interface descriptor and provider-session caching.
- Thinking behavior and unsupported-model fallback.
- LM Studio unload retry behavior.
- Data-only route diagnostics.

It does not cover prompt wording, route choice, JSON schema design, parser
semantics, embeddings, tool calls, streaming, batch calls, or native non-chat
provider APIs.

## Boundary Summary

```text
runtime module
  -> builds SystemMessage/HumanMessage
  -> owns LLMCallConfig from config.py
  -> LLInterface.ainvoke(..., config=...)
  -> backend descriptor and provider session
  -> provider adapter maps request fields
  -> provider-native chat model
  -> LLMResponse(content, backend, raw_response, usage)
```

Runtime modules own ordinary message content. The interface passes ordered
`BaseMessage` objects through without rewriting prompts, payloads, or message
order except when an enabled provider compatibility feature requires a
backend-specific control token. The current exception is Gemma 4 thinking,
where the OpenAI-compatible provider injects the Gemma thinking trigger on a
copied system message so caller-owned message objects are not mutated.

Provider-specific fields terminate inside provider adapters. Stage modules must
not construct `extra_body`, provider-native clients, backend-kind constants, or
model-family constants.

## Public Imports

Runtime modules import public contracts from the package facade:

```python
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
```

The exported public surface is:

- `LLInterface`
- `LLMCallConfig`
- `LLMThinkingConfig`
- `LLMResponse`
- `BackendDescriptor`
- `LLMInvoker`

Provider adapters, session cache internals, detection helpers, and reload
state are implementation details unless a test is specifically verifying that
boundary.

## Module-Owned Call Config

Each LLM-using module owns one or more immutable `LLMCallConfig` instances next
to the prompt and handler that consume them.

Config fields:

| Field | Owner | Meaning |
| --- | --- | --- |
| `stage_name` | caller | Stable Python stage/module identity for diagnostics. |
| `route_name` | caller | Route handle from `config.py`, such as `COGNITION_LLM`. |
| `base_url` | caller config | OpenAI-compatible endpoint base URL. |
| `api_key` | caller config | Provider credential. Hidden from dataclass repr. |
| `model` | caller config | Backend model name. |
| `temperature` | caller config | Provider-neutral generation setting. |
| `top_p` | caller config | Provider-neutral nucleus sampling setting. |
| `top_k` | caller config | Provider-neutral sampling setting retained in config/cache identity. |
| `max_completion_tokens` | caller config | Public completion budget field. |
| `presence_penalty` | caller config | Provider-neutral presence penalty when a stage uses it. |
| `thinking` | caller config | Boolean thinking request. Defaults to disabled. |

`max_tokens` is not part of this public interface. New route configuration must
use `max_completion_tokens`.

The caller chooses which config profile to pass. `LLInterface` only maps that
config into backend-compatible request fields.

## Public Invocation Contract

`LLInterface` exposes:

```python
class LLInterface:
    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse: ...

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
    ) -> LLMResponse: ...

    def describe_backend(
        self,
        *,
        config: LLMCallConfig,
    ) -> BackendDescriptor: ...

    def invalidate_backend(self, *, route_name: str | None = None) -> None: ...

    async def aclose(self) -> None: ...
```

`ainvoke` is the normal async path for runtime LLM stages. `invoke` exists for
sync JSON-repair usage. Both require an explicit `config=` keyword.

`LLMResponse.content` is the normalized text used by callers. `raw_response`
is retained for diagnostics or targeted tests, but production stage logic
should prefer the normalized fields unless it has a specific provider-aware
reason approved by this ICD.

For Gemma 4 thinking responses, normalized content removes raw
`<|channel>thought ... <channel|>` thought-channel spans when LM Studio exposes
them in the visible message content. The original provider response remains
available through `raw_response` for diagnostics.

## Backend Detection

Backend and model-family identity are detected by `LLInterface` from the
normalized model name. Runtime modules and `config.py` do not own backend kind
or model-family constants.

Current model-family labels include:

| Detected family | Model-name signal |
| --- | --- |
| `gemma4` | `gemma4` or `gemma-4` |
| `qwen` | `qwen` |
| `deepseek` | `deepseek` |
| `openai` | `gpt` or `openai` |
| `unknown` | no known signal |

The current backend kind is `openai_compatible`. Future native provider
support must extend the provider adapter boundary rather than leaking provider
fields into stage modules.

The detection module intentionally contains a TODO for future provider
capability probing. Runtime probing is not part of the current contract.

## Thinking Contract

Thinking is a boolean request owned by route config:

```python
LLMThinkingConfig(enabled=True)
```

Default behavior is disabled. When thinking is disabled, provider request
payloads must not contain thinking-specific fields.

Initial provider-side thinking support is limited to Gemma 4. When thinking is
enabled and the detected model family is `gemma4`, the OpenAI-compatible
provider maps it to both the request-level chat-template hint and Gemma's
prompt-level thinking trigger:

```python
extra_body = {
    "chat_template_kwargs": {"enable_thinking": True},
}
```

The provider also prefixes the first system message copy with `/think` when it
is not already present. If no system message is supplied, it inserts a new
leading system message containing `/think`. This is a backend compatibility
step for LM Studio/Gemma setups where `chat_template_kwargs.enable_thinking`
may be accepted but not enough to activate thinking.

When thinking is enabled for unsupported or unknown model families, the
interface records an `ignored_unsupported_model` strategy and omits
thinking-specific request fields and prompt triggers. It must not rewrite
ordinary prompt content to simulate thinking for unsupported models.

## Thinking Enablement Criteria

Thinking is a latency-expensive backend capability. Future agents should enable
it only when the stage benefits from hidden deliberation that is not already
captured by the stage's explicit prompt procedure or output contract.

Default to disabled when any of these are true:

- The prompt already externalizes the reasoning path as explicit staged output,
  audit steps, generation steps, or schema fields.
- The stage is on the live response path and runs for most user messages.
- The stage is a classifier, router, selector, extractor, validator, JSON
  repair call, or other bounded structured-output step.
- Deterministic code already owns the critical decision, such as permissions,
  persistence, limits, retry, cache invalidation, adapter delivery, or tool
  execution.
- The output is itself the inspectable reasoning artifact, such as cognition
  layer fields, boundary assessment, action selection, or evaluator feedback.

Consider enabling thinking when all of these are true:

- The stage is offline, background, operator-triggered, or otherwise outside
  the latency-critical live reply path.
- The stage performs open-ended synthesis, long-context comparison, difficult
  evidence reconciliation, or artifact generation where the prompt does not
  already expose the full reasoning path.
- The caller can tolerate higher completion latency and larger provider output.
- A real LLM comparison shows quality improvement that outweighs latency and
  token cost.
- The normalized output remains structurally valid after provider thinking
  output is stripped or separated.

For Kazusa's current architecture, the strongest "off" signal is an explicit
cognition chain or audit checklist. L1/L2/L2d/L3 cognition, boundary core,
dialog evaluation, RAG routing/extraction, relevance, decontextualization, and
JSON repair already have explicit contracts and should normally keep thinking
disabled. Better candidates are background text artifact generation, difficult
web evidence finalization, reflection synthesis, or global character-growth
candidate generation, and even those should be validated by live comparison
before changing defaults.

## Provider Adapter Contract

Provider adapters implement:

```python
class ProviderAdapter(Protocol):
    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> LLMResponse: ...

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: LLMCallConfig,
        backend: BackendDescriptor,
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...
```

The OpenAI-compatible provider is the only current production provider. Direct
`ChatOpenAI` construction belongs only inside
`providers/openai_compatible.py`.

Provider adapters own:

- provider-native client construction;
- request kwargs such as `model`, `base_url`, `api_key`, `temperature`,
  `top_p`, `max_completion_tokens`, `presence_penalty`, and provider-specific
  `extra_body`;
- conversion from provider-native responses to `LLMResponse`;
- provider-local chat-model cache identity.

Provider adapters must not change message content or ordering except for
documented backend compatibility controls such as the copied Gemma 4 thinking
trigger. Caller-owned message objects must not be mutated.

## Session And Cache Contract

Each `LLInterface()` instance owns its own:

- backend descriptor cache;
- provider session cache;
- provider-to-route ownership index;
- route generation counters.

This per-interface ownership allows future control-console backend switching
to invalidate one interface instance without relying on global singleton LLM
state.

`invalidate_backend(route_name=...)` invalidates descriptors and provider
sessions associated with that route. If a provider session is shared by
multiple routes, invalidating one associated route evicts the whole shared
session because provider-local chat-model caches are not route-partitioned.

`invalidate_backend(route_name=None)` clears all descriptor and provider
sessions for that interface instance.

Session cache keys and diagnostic fingerprints are separate. Raw API keys are
hashed for cache identity and must never appear in repr output, diagnostics,
route reports, logs, or test failure text.

## Reload Retry Contract

`ReloadingChatModel` wraps provider-native chat models and preserves the LM
Studio unload recovery behavior.

When a backend raises an OpenAI `BadRequestError` containing:

```text
The model has crashed without additional information.
```

the wrapper retries the same request once. Calls for the same normalized
`base_url` and model wait while the reload owner retry is active. Calls for a
different model key do not wait.

Other bad requests and non-unload errors are not retried by this path.

## Route Diagnostics

`llm_interface.diagnostics` exposes data-only route diagnostics used by
`llm_interface.route_report` for startup route reporting. Diagnostic rows
include:

- route name;
- backend kind;
- model name;
- normalized base URL;
- detected model family;
- thinking strategy;
- whether the route is required;
- whether the route has fallback-backed configuration.

The startup route report renders only operator-facing route identity:
`Route`, `Model`, `Source`, and `Optional Feature`. Optional features are
active, non-default capability tags such as `thinking_on`; disabled/default
states are not printed as feature tags.

Diagnostics and route reporting must not own route choice, prompt selection,
generation policy, or stage behavior. They must not expose raw API keys.

## Ownership Rules

Runtime modules own:

- prompt constants;
- rendered `SystemMessage` and `HumanMessage` content;
- message order;
- route selection;
- `LLMCallConfig` construction from `config.py`;
- parser and schema validation after receiving `LLMResponse.content`;
- semantic judgment about whether thinking should be enabled for a stage.

`llm_interface` owns:

- backend and model-family detection;
- provider adapter selection;
- provider-native client construction;
- provider request-field mapping;
- thinking payload mapping;
- documented provider-specific thinking triggers;
- response normalization;
- per-interface cache and invalidation;
- LM Studio unload retry coordination;
- API-key-safe diagnostics.

`llm_interface` does not own:

- prompt wording;
- evidence retrieval semantics;
- cognition or dialog policy;
- JSON repair semantics beyond sync invocation support;
- embeddings;
- streaming;
- tool calls;
- batch calls;
- native Anthropic API support.

## Compatibility Rules

- Adding optional fields to `LLMResponse` is compatible when existing callers
  can ignore them.
- Adding required fields to `LLMCallConfig` is breaking and requires all
  module-owned configs to migrate together.
- Renaming `max_completion_tokens` or adding public `max_tokens` is breaking
  for this ICD.
- Exposing provider-specific kwargs to runtime modules is breaking.
- Reintroducing `get_llm()` or a compatibility wrapper that preserves the old
  `.ainvoke(messages)` call shape is breaking.
- Moving direct `ChatOpenAI` construction outside the provider adapter is
  breaking.
- Adding a native provider family requires updating this ICD, provider tests,
  route diagnostics, and backend detection behavior.

## Failure Behavior

Unsupported backend kinds fail with `ValueError` before provider invocation.

Unsupported thinking requests do not fail. They are ignored by omitting
thinking payload fields and returning a backend descriptor with
`thinking_strategy="ignored_unsupported_model"`.

Provider failures propagate unless the error matches the LM Studio unload
retry signature. Parser failures remain owned by the caller after
`LLMResponse.content` is returned.

## Test Contract

Required deterministic coverage includes:

- `LLMCallConfig` uses `max_completion_tokens` and redacts `api_key`;
- backend/model-family detection;
- Gemma 4 thinking payload mapping;
- Gemma 4 thinking prompt-trigger mapping without caller message mutation;
- unsupported-model thinking ignore behavior;
- provider request mapping and message pass-through;
- per-interface descriptor caching and route invalidation;
- LM Studio reload retry and waiter behavior;
- static migration checks for removed `get_llm()` and provider-boundary
  imports;
- representative message/config equivalence for migrated call sites.

Live LLM tests remain behavioral smoke evidence. They must be run one case at
a time and inspected according to the project test contract.
