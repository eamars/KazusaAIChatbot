from unittest.mock import AsyncMock

import pytest

from scripts import ensure_vector_search_indexes as script_module
from kazusa_ai_chatbot.db import script_operations


@pytest.mark.asyncio
async def test_vector_search_index_dry_run_reports_missing_fields_without_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_index = AsyncMock(
        return_value={
            "collection": "conversation_history",
            "index_name": "conversation_history_vector_index",
            "status": "missing_filter_paths",
            "requires_recreate": True,
            "missing_filter_paths": ["platform_channel_id"],
        }
    )
    apply_index = AsyncMock()
    monkeypatch.setattr(script_module, "inspect_vector_search_index", inspect_index)
    monkeypatch.setattr(script_module, "apply_vector_search_index", apply_index)

    result = await script_module.ensure_vector_search_indexes(
        collections=["conversation_history"],
        apply=False,
        wait_ready=False,
    )

    assert result["apply"] is False
    assert result["indexes"][0]["requires_recreate"] is True
    inspect_index.assert_awaited_once_with("conversation_history")
    apply_index.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_search_index_apply_recreates_missing_filter_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_index = AsyncMock(
        return_value={
            "collection": "conversation_history",
            "index_name": "conversation_history_vector_index",
            "status": "missing_filter_paths",
            "requires_recreate": True,
            "missing_filter_paths": ["platform_channel_id"],
        }
    )
    apply_index = AsyncMock(
        return_value={
            "collection": "conversation_history",
            "index_name": "conversation_history_vector_index",
            "status": "applied",
            "requires_recreate": False,
            "missing_filter_paths": [],
        }
    )
    monkeypatch.setattr(script_module, "inspect_vector_search_index", inspect_index)
    monkeypatch.setattr(script_module, "apply_vector_search_index", apply_index)

    result = await script_module.ensure_vector_search_indexes(
        collections=["conversation_history"],
        apply=True,
        wait_ready=True,
    )

    assert result["apply"] is True
    assert result["indexes"][0]["status"] == "applied"
    inspect_index.assert_awaited_once_with("conversation_history")
    apply_index.assert_awaited_once_with("conversation_history", wait_ready=True)


@pytest.mark.asyncio
async def test_vector_search_index_apply_prepares_model_before_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Collection:
        async def drop_search_index(self, index_name: str) -> None:
            events.append(f"drop:{index_name}")

        async def create_search_index(self, search_index_model: object) -> None:
            events.append(f"create:{search_index_model}")

    class _Db:
        def __getitem__(self, collection_name: str) -> _Collection:
            return _Collection()

    async def get_embedding(text: str) -> list[float]:
        events.append(f"embed:{text}")
        return [0.1, 0.2, 0.3]

    def build_model(**kwargs: object) -> str:
        events.append(f"build:{kwargs['index_name']}")
        return "search-index-model"

    monkeypatch.setattr(script_operations, "get_db", AsyncMock(return_value=_Db()))
    monkeypatch.setattr(
        script_operations,
        "_find_search_index",
        AsyncMock(return_value={"name": "conversation_history_vector_index"}),
    )
    monkeypatch.setattr(script_operations, "get_text_embedding", get_embedding)
    monkeypatch.setattr(script_operations, "build_vector_search_index_model", build_model)

    await script_operations.apply_vector_search_index(
        "conversation_history",
        wait_ready=False,
    )

    assert events == [
        "embed:test",
        "build:conversation_history_vector_index",
        "drop:conversation_history_vector_index",
        "create:search-index-model",
    ]


@pytest.mark.asyncio
async def test_inspect_vector_search_index_reports_definition_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_document = {
        "latestDefinition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "wrong_embedding",
                    "numDimensions": 512,
                    "similarity": "dotProduct",
                },
                {"type": "filter", "path": "platform"},
            ]
        }
    }
    monkeypatch.setattr(
        script_operations,
        "_find_search_index",
        AsyncMock(return_value=index_document),
    )
    monkeypatch.setattr(
        script_operations,
        "get_text_embedding",
        AsyncMock(return_value=[0.1, 0.2, 0.3]),
    )

    result = await script_operations.inspect_vector_search_index(
        "conversation_history"
    )

    assert result["status"] == "definition_mismatch"
    assert result["requires_recreate"] is True
    assert result["definition_issues"] == [
        "vector_path",
        "num_dimensions",
        "similarity",
        "missing_filter_path:platform_channel_id",
        "missing_filter_path:global_user_id",
        "missing_filter_path:role",
        "missing_filter_path:timestamp",
    ]
