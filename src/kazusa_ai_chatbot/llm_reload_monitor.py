"""Monitor LM Studio unload recovery for shared chat model clients."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from openai import BadRequestError

ModelKey = tuple[str, str]

_LM_STUDIO_MODEL_UNLOAD_SIGNATURE = (
    "The model has crashed without additional information."
)


@dataclass
class _ReloadState:
    """Track the active reload owner for one configured model key."""

    finished: threading.Event = field(default_factory=threading.Event)


_reload_lock = threading.Lock()
_active_reloads: dict[ModelKey, _ReloadState] = {}


def is_lm_studio_model_unload_error(exc: BaseException) -> bool:
    """Return whether an exception contains the confirmed LM Studio unload text.

    Args:
        exc: Exception raised by an OpenAI-compatible chat client.

    Returns:
        Whether the exception text contains the exact confirmed unload
        signature and should receive the one-shot reload retry.
    """

    return_value = _LM_STUDIO_MODEL_UNLOAD_SIGNATURE in str(exc)
    return return_value


def _active_reload_state(model_key: ModelKey) -> _ReloadState | None:
    """Return the current reload owner state for a model key, if one exists."""

    with _reload_lock:
        reload_state = _active_reloads.get(model_key)
    return reload_state


def _claim_reload_owner(model_key: ModelKey) -> tuple[_ReloadState, bool]:
    """Create or return the active reload owner for a model key.

    Args:
        model_key: Normalized OpenAI-compatible endpoint and model name.

    Returns:
        The active reload state and whether the caller owns the reload retry.
    """

    with _reload_lock:
        reload_state = _active_reloads.get(model_key)
        if reload_state is not None:
            return_value = (reload_state, False)
            return return_value

        reload_state = _ReloadState()
        _active_reloads[model_key] = reload_state

    return_value = (reload_state, True)
    return return_value


def _release_reload_owner(
    model_key: ModelKey,
    reload_state: _ReloadState,
) -> None:
    """Release waiters for a completed reload owner attempt."""

    with _reload_lock:
        current_reload_state = _active_reloads.get(model_key)
        if current_reload_state is reload_state:
            del _active_reloads[model_key]
        reload_state.finished.set()


def _wait_for_reload_state(reload_state: _ReloadState) -> None:
    """Block the current sync caller until the reload owner completes."""

    reload_state.finished.wait()


async def _wait_for_reload_state_async(reload_state: _ReloadState) -> None:
    """Wait for the reload owner without blocking the event loop."""

    await asyncio.to_thread(reload_state.finished.wait)


def _wait_for_no_active_reload(model_key: ModelKey) -> None:
    """Block sync callers while the same model key is in unload recovery."""

    while True:
        reload_state = _active_reload_state(model_key)
        if reload_state is None:
            return
        _wait_for_reload_state(reload_state)


async def _wait_for_no_active_reload_async(model_key: ModelKey) -> None:
    """Wait async callers while the same model key is in unload recovery."""

    while True:
        reload_state = _active_reload_state(model_key)
        if reload_state is None:
            return
        await _wait_for_reload_state_async(reload_state)


class MonitoredChatModel:
    """Wrap a chat model with one retry for confirmed LM Studio unloads.

    Args:
        inner_llm: Chat client exposing ``ainvoke`` and ``invoke``.
        base_url: OpenAI-compatible endpoint used for model-key scoping.
        model: Configured model name used for model-key scoping.
    """

    def __init__(
        self,
        inner_llm: object,
        *,
        base_url: str,
        model: str,
    ) -> None:
        self._inner_llm = inner_llm
        self._model_key: ModelKey = (base_url.rstrip("/"), model)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes to the wrapped chat model."""

        delegated_value = getattr(self._inner_llm, name)
        return delegated_value

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped async chat model with unload recovery."""

        await _wait_for_no_active_reload_async(self._model_key)
        try:
            response = await self._inner_llm.ainvoke(*args, **kwargs)
        except BadRequestError as exc:
            if not is_lm_studio_model_unload_error(exc):
                raise
            response = await self._retry_ainvoke_after_unload(args, kwargs)

        return response

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped sync chat model with unload recovery."""

        _wait_for_no_active_reload(self._model_key)
        try:
            response = self._inner_llm.invoke(*args, **kwargs)
        except BadRequestError as exc:
            if not is_lm_studio_model_unload_error(exc):
                raise
            response = self._retry_invoke_after_unload(args, kwargs)

        return response

    async def _retry_ainvoke_after_unload(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Retry one async call after the first confirmed unload error."""

        reload_state, is_owner = _claim_reload_owner(self._model_key)
        if not is_owner:
            await _wait_for_reload_state_async(reload_state)
            response = await self._inner_llm.ainvoke(*args, **kwargs)
            return response

        try:
            response = await self._inner_llm.ainvoke(*args, **kwargs)
        finally:
            _release_reload_owner(self._model_key, reload_state)

        return response

    def _retry_invoke_after_unload(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Retry one sync call after the first confirmed unload error."""

        reload_state, is_owner = _claim_reload_owner(self._model_key)
        if not is_owner:
            _wait_for_reload_state(reload_state)
            response = self._inner_llm.invoke(*args, **kwargs)
            return response

        try:
            response = self._inner_llm.invoke(*args, **kwargs)
        finally:
            _release_reload_owner(self._model_key, reload_state)

        return response


def monitored_chat_model(
    inner_llm: object,
    *,
    base_url: str,
    model: str,
) -> MonitoredChatModel:
    """Wrap a chat model with shared unload monitoring.

    Args:
        inner_llm: Chat client to invoke after same-model recovery pauses.
        base_url: OpenAI-compatible endpoint used for model-key scoping.
        model: Configured model name used for model-key scoping.

    Returns:
        A chat-model wrapper exposing ``ainvoke``, ``invoke``, and delegated
        attributes from the wrapped model.
    """

    monitored_model = MonitoredChatModel(
        inner_llm,
        base_url=base_url,
        model=model,
    )
    return monitored_model
