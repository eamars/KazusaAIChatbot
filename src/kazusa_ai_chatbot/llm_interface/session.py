"""Per-interface backend descriptor and provider session caching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from kazusa_ai_chatbot.llm_interface.contracts import (
    BackendDescriptor,
    LLMCallConfig,
    ProviderAdapter,
)
from kazusa_ai_chatbot.llm_interface.detection import (
    detect_backend_descriptor,
    normalize_base_url,
)
from kazusa_ai_chatbot.llm_interface.providers.openai_compatible import (
    OpenAICompatibleProvider,
)

SessionCacheKey = tuple[str, str, str, str]
DiagnosticFingerprint = tuple[object, ...]


@dataclass
class InterfaceSessionCache:
    """Own per-interface descriptors, provider sessions, and generations."""

    _descriptors: dict[tuple[DiagnosticFingerprint, int], BackendDescriptor] = (
        field(default_factory=dict)
    )
    _providers: dict[SessionCacheKey, ProviderAdapter] = field(default_factory=dict)
    _provider_routes: dict[SessionCacheKey, set[str]] = field(default_factory=dict)
    _generations: dict[str, int] = field(default_factory=dict)

    def describe_backend(self, *, config: LLMCallConfig) -> BackendDescriptor:
        """Return a cached descriptor for a config and route generation."""

        generation = self._generations.get(config.route_name, 1)
        descriptor_key = _descriptor_key(config=config, generation=generation)
        cached_descriptor = self._descriptors.get(descriptor_key)
        if cached_descriptor is not None:
            return cached_descriptor

        descriptor = detect_backend_descriptor(
            config=config,
            generation=generation,
        )
        self._descriptors[descriptor_key] = descriptor
        return descriptor

    def provider_for(
        self,
        *,
        config: LLMCallConfig,
        descriptor: BackendDescriptor,
    ) -> ProviderAdapter:
        """Return the provider session for a backend descriptor."""

        session_key = build_session_cache_key(
            config=config,
            descriptor=descriptor,
        )
        provider = self._providers.get(session_key)
        if provider is not None:
            self._provider_routes.setdefault(session_key, set()).add(
                config.route_name
            )
            return provider

        if descriptor.backend_kind != "openai_compatible":
            raise ValueError(
                f"Unsupported LLM backend kind: {descriptor.backend_kind}"
            )

        provider = OpenAICompatibleProvider()
        self._providers[session_key] = provider
        self._provider_routes[session_key] = {config.route_name}
        return provider

    def invalidate_backend(self, *, route_name: str | None = None) -> None:
        """Invalidate descriptors and sessions for one route or all routes."""

        if route_name is None:
            for cached_route in list(self._generations):
                self._generations[cached_route] += 1
            self._descriptors.clear()
            self._providers.clear()
            self._provider_routes.clear()
            return

        current_generation = self._generations.get(route_name, 1)
        self._generations[route_name] = current_generation + 1
        self._descriptors = {
            descriptor_key: descriptor
            for descriptor_key, descriptor in self._descriptors.items()
            if descriptor.route_name != route_name
        }
        kept_providers: dict[SessionCacheKey, ProviderAdapter] = {}
        kept_provider_routes: dict[SessionCacheKey, set[str]] = {}
        for session_key, provider in self._providers.items():
            route_names = self._provider_routes[session_key]
            if route_name in route_names:
                continue
            kept_providers[session_key] = provider
            kept_provider_routes[session_key] = route_names

        self._providers = kept_providers
        self._provider_routes = kept_provider_routes

    async def aclose(self) -> None:
        """Close cached provider sessions and clear local cache state."""

        providers = list(self._providers.values())
        self._providers.clear()
        self._provider_routes.clear()
        self._descriptors.clear()
        for provider in providers:
            await provider.aclose()


def api_key_hash(api_key: str) -> str:
    """Return a stable non-secret API key identity."""

    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest


def build_session_cache_key(
    *,
    config: LLMCallConfig,
    descriptor: BackendDescriptor,
) -> SessionCacheKey:
    """Build the provider session cache key for one interface instance."""

    session_key = (
        descriptor.normalized_base_url,
        api_key_hash(config.api_key),
        config.model,
        descriptor.backend_kind,
    )
    return session_key


def build_diagnostic_fingerprint(
    *,
    config: LLMCallConfig,
    descriptor: BackendDescriptor,
) -> DiagnosticFingerprint:
    """Build a diagnostic fingerprint that excludes raw API keys."""

    fingerprint = (
        config.stage_name,
        config.route_name,
        descriptor.normalized_base_url,
        config.model,
        descriptor.backend_kind,
        descriptor.model_family,
        config.temperature,
        config.top_p,
        config.top_k,
        config.max_completion_tokens,
        config.presence_penalty,
        config.timeout_seconds,
        config.thinking.enabled,
        descriptor.thinking_strategy,
    )
    return fingerprint


def _descriptor_key(
    *,
    config: LLMCallConfig,
    generation: int,
) -> tuple[DiagnosticFingerprint, int]:
    """Build a descriptor cache key without requiring an existing descriptor."""

    descriptor = detect_backend_descriptor(
        config=config,
        generation=generation,
    )
    fingerprint = build_diagnostic_fingerprint(
        config=config,
        descriptor=descriptor,
    )
    descriptor_key = (fingerprint, generation)
    return descriptor_key

