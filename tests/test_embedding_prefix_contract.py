from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.db import _client as module


class _FakeEmbeddingClient:
    """Capture embedding inputs sent through the adapter."""

    def __init__(self, *, reverse_data: bool = False) -> None:
        self.inputs: list[list[str]] = []
        self.embeddings = self
        self.reverse_data = reverse_data

    async def create(self, *, input: list[str], model: str):
        """Return deterministic vectors while recording effective input text."""

        self.inputs.append(list(input))
        data = [
            SimpleNamespace(index=index, embedding=[float(index), float(len(text))])
            for index, text in enumerate(input)
        ]
        if self.reverse_data:
            data = list(reversed(data))
        response = SimpleNamespace(data=data)
        return response


def _patch_embedding_client(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    *,
    reverse_data: bool = False,
) -> _FakeEmbeddingClient:
    """Patch the configured embedding client for adapter contract tests."""

    fake_client = _FakeEmbeddingClient(reverse_data=reverse_data)
    monkeypatch.setattr(module, "EMBEDDING_MODEL", model)
    monkeypatch.setattr(module, "_get_embed_client", lambda: fake_client)
    return fake_client


def test_get_text_embedding_signature_remains_single_text_argument() -> None:
    """The compatibility helper must keep its public call signature."""

    signature = inspect.signature(module.get_text_embedding)

    assert list(signature.parameters) == ["text"]


@pytest.mark.asyncio
async def test_nomic_query_embedding_adds_search_query_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nomic query embeddings should use the query task instruction."""

    fake_client = _patch_embedding_client(
        monkeypatch,
        "text-embedding-nomic-embed-text-v2-moe",
    )

    await module.get_query_text_embedding("gpu market share")

    assert fake_client.inputs == [["search_query: gpu market share"]]


@pytest.mark.asyncio
async def test_nomic_document_embedding_adds_search_document_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nomic document embeddings should use the document task instruction."""

    fake_client = _patch_embedding_client(
        monkeypatch,
        "text-embedding-nomic-embed-text-v2-moe",
    )

    await module.get_document_text_embedding("Steam GPU row")

    assert fake_client.inputs == [["search_document: Steam GPU row"]]


@pytest.mark.asyncio
async def test_non_nomic_embedding_does_not_add_nomic_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Nomic models should preserve raw input until separately approved."""

    fake_client = _patch_embedding_client(monkeypatch, "other-embedding-model")

    await module.get_query_text_embedding("gpu market share")
    await module.get_document_text_embedding("Steam GPU row")

    assert fake_client.inputs == [["gpu market share"], ["Steam GPU row"]]


@pytest.mark.asyncio
async def test_batch_document_embeddings_prefixes_raw_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch document embedding should treat prefix-like text as content."""

    fake_client = _patch_embedding_client(
        monkeypatch,
        "text-embedding-nomic-embed-text-v2-moe",
    )

    await module.get_document_text_embeddings_batch([
        "first row",
        "search_document: second row",
    ])

    assert fake_client.inputs == [[
        "search_document: first row",
        "search_document: search_document: second row",
    ]]


@pytest.mark.asyncio
async def test_batch_embeddings_preserve_input_order_from_response_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch embeddings should be ordered by provider response index."""

    fake_client = _patch_embedding_client(
        monkeypatch,
        "text-embedding-nomic-embed-text-v2-moe",
        reverse_data=True,
    )

    embeddings = await module.get_document_text_embeddings_batch([
        "first row",
        "second row",
    ])

    assert fake_client.inputs == [[
        "search_document: first row",
        "search_document: second row",
    ]]
    assert embeddings == [
        [0.0, 26.0],
        [1.0, 27.0],
    ]


@pytest.mark.asyncio
async def test_embedding_prefixing_keeps_prefix_like_content_transparent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw content that starts with a prefix string should still be role-shaped."""

    fake_client = _patch_embedding_client(
        monkeypatch,
        "text-embedding-nomic-embed-text-v2-moe",
    )

    await module.get_query_text_embedding("search_query: gpu market share")
    await module.get_document_text_embedding("search_document: Steam GPU row")

    assert fake_client.inputs == [
        ["search_query: search_query: gpu market share"],
        ["search_document: search_document: Steam GPU row"],
    ]


@pytest.mark.asyncio
async def test_embedding_prefixing_adds_requested_role_to_opposite_prefix_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requested embedding role should not be inferred from raw text."""

    fake_client = _patch_embedding_client(
        monkeypatch,
        "text-embedding-nomic-embed-text-v2-moe",
    )

    await module.get_query_text_embedding("search_document: stored row")
    await module.get_document_text_embedding("search_query: search intent")

    assert fake_client.inputs == [
        ["search_query: search_document: stored row"],
        ["search_document: search_query: search intent"],
    ]
