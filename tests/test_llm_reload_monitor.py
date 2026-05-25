import asyncio

import httpx
import openai
import pytest

from kazusa_ai_chatbot.llm_reload_monitor import monitored_chat_model


_UNLOAD_ERROR_TEXT = (
    "Error code: 400 - {'error': 'The model has crashed without additional "
    "information. (Exit code: 18446744072635812000)'}"
)


class _FakeResponse:
    """Small response object for monitor tests."""

    def __init__(self, content: str) -> None:
        self.content = content


class _AsyncSequenceLLM:
    """Fake async LLM that returns or raises scripted outcomes."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _SyncSequenceLLM:
    """Fake sync LLM that returns or raises scripted outcomes."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def invoke(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _OwnerAsyncLLM:
    """Fake LLM whose retry waits until the test releases it."""

    def __init__(self) -> None:
        self.calls = 0
        self.retry_started = asyncio.Event()
        self.release_retry = asyncio.Event()

    async def ainvoke(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise _unload_error()

        self.retry_started.set()
        await self.release_retry.wait()
        response = _FakeResponse("owner retry")
        return response


class _FailingOwnerAsyncLLM:
    """Fake LLM whose retry fails after exposing the active reload state."""

    def __init__(self) -> None:
        self.calls = 0
        self.retry_started = asyncio.Event()
        self.release_retry = asyncio.Event()

    async def ainvoke(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise _unload_error()

        self.retry_started.set()
        await self.release_retry.wait()
        raise RuntimeError("reload retry failed")


class _ObserverAsyncLLM:
    """Fake LLM used to verify same-key wait and different-key independence."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        response = _FakeResponse("observer")
        return response


class _ConcurrentUnloadCoordinator:
    """Coordinate two fake LLMs that unload on their first attempt together."""

    def __init__(self) -> None:
        self.first_attempt_count = 0
        self.retry_call_count = 0
        self.first_attempts_ready = asyncio.Event()
        self.release_first_attempts = asyncio.Event()
        self.retry_started = asyncio.Event()
        self.release_retry = asyncio.Event()


class _RacingUnloadAsyncLLM:
    """Fake LLM that joins a two-call unload race before retrying."""

    def __init__(
        self,
        name: str,
        coordinator: _ConcurrentUnloadCoordinator,
    ) -> None:
        self.name = name
        self.coordinator = coordinator
        self.calls = 0

    async def ainvoke(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            self.coordinator.first_attempt_count += 1
            if self.coordinator.first_attempt_count == 2:
                self.coordinator.first_attempts_ready.set()
            await self.coordinator.release_first_attempts.wait()
            raise _unload_error()

        self.coordinator.retry_call_count += 1
        self.coordinator.retry_started.set()
        await self.coordinator.release_retry.wait()
        response = _FakeResponse(f"{self.name} retry")
        return response


def _bad_request_error(message: str) -> openai.BadRequestError:
    """Build an OpenAI bad-request error with the requested message."""

    request = httpx.Request("POST", "http://localhost:1234/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={"error": message},
    )
    error = openai.BadRequestError(
        message,
        response=response,
        body={"error": message},
    )
    return error


def _unload_error() -> openai.BadRequestError:
    """Build the confirmed LM Studio unload error."""

    error = _bad_request_error(_UNLOAD_ERROR_TEXT)
    return error


@pytest.mark.asyncio
async def test_async_unload_error_retries_same_call_once() -> None:
    llm = _AsyncSequenceLLM([
        _unload_error(),
        _FakeResponse("reloaded"),
    ])
    monitored = monitored_chat_model(
        llm,
        base_url="http://localhost:1234/v1/",
        model="async-retry-success",
    )

    response = await monitored.ainvoke(["message"], request_id="abc")

    assert response.content == "reloaded"
    assert len(llm.calls) == 2
    assert llm.calls[0] == ((["message"],), {"request_id": "abc"})
    assert llm.calls[1] == ((["message"],), {"request_id": "abc"})


@pytest.mark.asyncio
async def test_async_unload_retry_failure_raises_and_releases_waiters() -> None:
    owner_llm = _FailingOwnerAsyncLLM()
    observer_llm = _ObserverAsyncLLM()
    owner = monitored_chat_model(
        owner_llm,
        base_url="http://localhost:1234/v1",
        model="async-retry-failure-release",
    )
    observer = monitored_chat_model(
        observer_llm,
        base_url="http://localhost:1234/v1/",
        model="async-retry-failure-release",
    )

    owner_task = asyncio.create_task(owner.ainvoke(["owner"]))
    await asyncio.wait_for(owner_llm.retry_started.wait(), timeout=1.0)
    observer_task = asyncio.create_task(observer.ainvoke(["observer"]))
    await asyncio.sleep(0)

    assert observer_llm.calls == 0
    owner_llm.release_retry.set()
    with pytest.raises(RuntimeError, match="reload retry failed"):
        await asyncio.wait_for(owner_task, timeout=1.0)
    observer_response = await asyncio.wait_for(observer_task, timeout=1.0)

    assert observer_response.content == "observer"
    assert observer_llm.calls == 1


@pytest.mark.asyncio
async def test_async_non_unload_error_is_not_retried() -> None:
    llm = _AsyncSequenceLLM([
        _bad_request_error("ordinary bad request"),
    ])
    monitored = monitored_chat_model(
        llm,
        base_url="http://localhost:1234/v1",
        model="async-non-unload",
    )

    with pytest.raises(openai.BadRequestError, match="ordinary bad request"):
        await monitored.ainvoke(["message"])

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_async_same_model_call_waits_during_owner_retry() -> None:
    owner_llm = _OwnerAsyncLLM()
    observer_llm = _ObserverAsyncLLM()
    owner = monitored_chat_model(
        owner_llm,
        base_url="http://localhost:1234/v1",
        model="async-same-model-wait",
    )
    observer = monitored_chat_model(
        observer_llm,
        base_url="http://localhost:1234/v1/",
        model="async-same-model-wait",
    )

    owner_task = asyncio.create_task(owner.ainvoke(["owner"]))
    await asyncio.wait_for(owner_llm.retry_started.wait(), timeout=1.0)
    observer_task = asyncio.create_task(observer.ainvoke(["observer"]))
    await asyncio.sleep(0)

    assert observer_llm.calls == 0
    owner_llm.release_retry.set()
    owner_response = await asyncio.wait_for(owner_task, timeout=1.0)
    observer_response = await asyncio.wait_for(observer_task, timeout=1.0)

    assert owner_response.content == "owner retry"
    assert observer_response.content == "observer"
    assert observer_llm.calls == 1


@pytest.mark.asyncio
async def test_async_different_model_call_does_not_wait() -> None:
    owner_llm = _OwnerAsyncLLM()
    observer_llm = _ObserverAsyncLLM()
    owner = monitored_chat_model(
        owner_llm,
        base_url="http://localhost:1234/v1",
        model="async-owner-model",
    )
    observer = monitored_chat_model(
        observer_llm,
        base_url="http://localhost:1234/v1",
        model="async-observer-model",
    )

    owner_task = asyncio.create_task(owner.ainvoke(["owner"]))
    await asyncio.wait_for(owner_llm.retry_started.wait(), timeout=1.0)
    observer_response = await asyncio.wait_for(
        observer.ainvoke(["observer"]),
        timeout=1.0,
    )

    assert observer_response.content == "observer"
    assert observer_llm.calls == 1
    owner_llm.release_retry.set()
    owner_response = await asyncio.wait_for(owner_task, timeout=1.0)
    assert owner_response.content == "owner retry"


@pytest.mark.asyncio
async def test_async_concurrent_unload_uses_one_owner() -> None:
    coordinator = _ConcurrentUnloadCoordinator()
    first_llm = _RacingUnloadAsyncLLM("first", coordinator)
    second_llm = _RacingUnloadAsyncLLM("second", coordinator)
    owner = monitored_chat_model(
        first_llm,
        base_url="http://localhost:1234/v1",
        model="async-concurrent-unload",
    )
    observer = monitored_chat_model(
        second_llm,
        base_url="http://localhost:1234/v1",
        model="async-concurrent-unload",
    )

    owner_task = asyncio.create_task(owner.ainvoke(["owner"]))
    observer_task = asyncio.create_task(observer.ainvoke(["observer"]))
    await asyncio.wait_for(coordinator.first_attempts_ready.wait(), timeout=1.0)
    coordinator.release_first_attempts.set()
    await asyncio.wait_for(coordinator.retry_started.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert coordinator.retry_call_count == 1
    coordinator.release_retry.set()
    owner_response = await asyncio.wait_for(owner_task, timeout=1.0)
    observer_response = await asyncio.wait_for(observer_task, timeout=1.0)

    response_contents = {owner_response.content, observer_response.content}
    assert response_contents == {"first retry", "second retry"}
    assert first_llm.calls == 2
    assert second_llm.calls == 2
    assert coordinator.retry_call_count == 2


def test_sync_unload_error_retries_same_call_once() -> None:
    llm = _SyncSequenceLLM([
        _unload_error(),
        _FakeResponse("sync reloaded"),
    ])
    monitored = monitored_chat_model(
        llm,
        base_url="http://localhost:1234/v1/",
        model="sync-retry-success",
    )

    response = monitored.invoke(["message"], request_id="abc")

    assert response.content == "sync reloaded"
    assert len(llm.calls) == 2
    assert llm.calls[0] == ((["message"],), {"request_id": "abc"})
    assert llm.calls[1] == ((["message"],), {"request_id": "abc"})
