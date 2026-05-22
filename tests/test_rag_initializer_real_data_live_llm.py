"""Real-data live LLM checks for RAG initializer recent-context behavior."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from experiments.conversation_graph_poc import build_graph, project_graph_context
from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_initializer as initializer_module
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc
from kazusa_ai_chatbot.utils import trim_history_dict
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_EXPORT_PATH = (
    PROJECT_ROOT
    / "test_artifacts"
    / "conversation_graph_priority"
    / "qq_group_905393941_latest300.json"
)
PRIVATE_EXPORT_PATH = (
    PROJECT_ROOT
    / "test_artifacts"
    / "conversation_graph_priority"
    / "qq_private_673225019_latest300.json"
)
CURRENT_RECENT_LIMIT = 5
CURRENT_WIDE_LIMIT = 10
EXPANDED_WINDOW_LIMIT = 50


class _NoCacheRuntime:
    """Disable initializer cache reads and writes for live prompt inspection."""

    async def get(self, *args: object, **kwargs: object) -> None:
        """Return no cached value so every mode reaches the live LLM."""

        del args, kwargs
        return None

    async def store(self, *args: object, **kwargs: object) -> None:
        """Ignore cache writes from the production initializer."""

        del args, kwargs


class _CapturingInitializerLLM:
    """Record production initializer prompts and raw model responses."""

    def __init__(self, wrapped: Any) -> None:
        """Wrap the configured initializer LLM.

        Args:
            wrapped: Existing LangChain-compatible chat model.
        """

        self._wrapped = wrapped
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        """Forward the call while preserving prompt and response content."""

        call_record = {
            "messages": [
                {
                    "type": message.__class__.__name__,
                    "content": getattr(message, "content", ""),
                }
                for message in messages
            ]
        }
        response = await self._wrapped.ainvoke(messages)
        call_record["raw_output"] = getattr(response, "content", "")
        self.calls.append(call_record)
        return response


async def _noop_async(*args: object, **kwargs: object) -> None:
    """Ignore persistent Cache2 maintenance during live prompt tests."""

    del args, kwargs


async def _skip_if_llm_unavailable() -> None:
    """Skip these live checks when the configured planner route is offline."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{RAG_PLANNER_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {RAG_PLANNER_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{RAG_PLANNER_LLM_BASE_URL}"
        )


def _load_export_messages(path: Path) -> list[dict[str, Any]]:
    """Read one exported conversation-history artifact.

    Args:
        path: JSON export created by the project chat-history export script.

    Returns:
        Chronological conversation-history rows.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    messages = [
        row
        for row in document["messages"]
        if isinstance(row, dict)
    ]
    return messages


def _find_message_index(messages: list[dict[str, Any]], message_id: str) -> int:
    """Find the index of a platform message in exported rows."""

    for index, row in enumerate(messages):
        if str(row.get("platform_message_id", "")) == message_id:
            return index
    raise AssertionError(f"message id not found in export: {message_id}")


def _message_text_for_graph_node(node: dict[str, Any]) -> str:
    """Return the full graph node text, including attachment descriptions."""

    node_text = str(node.get("text", "")).strip()
    return node_text


def _graph_history(
    messages: list[dict[str, Any]],
    *,
    anchor_message_id: str,
) -> list[dict[str, Any]]:
    """Build a graph-selected prompt history for one anchor turn."""

    graph = build_graph(messages)
    graph_context = project_graph_context(
        graph,
        anchor_message_id=anchor_message_id,
    )
    full_text_by_message_id = {
        str(node.get("platform_message_id", "")): _message_text_for_graph_node(node)
        for node in graph
    }
    history_rows: list[dict[str, Any]] = []
    active_messages = graph_context["active_thread_messages"]
    for graph_row in active_messages:
        if graph_row.get("is_current_turn") is True:
            continue
        platform_message_id = str(graph_row.get("platform_message_id", ""))
        body_text = full_text_by_message_id.get(platform_message_id, "")
        history_row = {
            "name": graph_row["speaker"],
            "display_name": graph_row["speaker"],
            "platform_message_id": platform_message_id,
            "role": graph_row["role"],
            "body_text": body_text,
            "timestamp": graph_row["timestamp"],
            "conversation_graph_relation": graph_row["relation_to_anchor"],
            "conversation_graph_edge_reasons": graph_row["edge_reasons"],
        }
        history_rows.append(history_row)

    if not history_rows:
        nearby_messages = graph_context["nearby_ambient_messages"]
        for graph_row in nearby_messages:
            platform_message_id = str(graph_row.get("platform_message_id", ""))
            body_text = full_text_by_message_id.get(platform_message_id, "")
            history_rows.append(
                {
                    "name": graph_row["speaker"],
                    "display_name": graph_row["speaker"],
                    "platform_message_id": platform_message_id,
                    "role": graph_row["role"],
                    "body_text": body_text,
                    "timestamp": graph_row["timestamp"],
                    "conversation_graph_relation": graph_row["relation_to_anchor"],
                    "conversation_graph_edge_reasons": graph_row["edge_reasons"],
                }
            )

    return history_rows


def _character_name(messages: list[dict[str, Any]]) -> str:
    """Return the active character name from assistant rows when available."""

    for row in messages:
        if row.get("role") == "assistant" and row.get("display_name"):
            character_name = str(row["display_name"])
            return character_name
    character_name = "the active character"
    return character_name


def _prompt_message_context(anchor: dict[str, Any]) -> dict[str, Any]:
    """Build the current-turn prompt message context from an exported row."""

    context = {
        "body_text": str(anchor.get("body_text") or ""),
        "mentions": anchor.get("mentions") or [],
        "attachments": anchor.get("attachments") or [],
        "addressed_to_global_user_ids": anchor["addressed_to_global_user_ids"],
        "broadcast": bool(anchor["broadcast"]),
        "reply": anchor.get("reply_context") or {},
    }
    return context


def _base_runtime_context(
    *,
    anchor: dict[str, Any],
    history_recent: list[dict[str, Any]],
    history_wide: list[dict[str, Any]],
    context_source: str,
) -> dict[str, Any]:
    """Build a production-shaped RAG initializer runtime context."""

    timestamp = str(anchor["timestamp"])
    context = {
        "platform": anchor["platform"],
        "platform_channel_id": anchor["platform_channel_id"],
        "channel_type": anchor["channel_type"],
        "global_user_id": anchor.get("global_user_id", ""),
        "platform_user_id": anchor.get("platform_user_id", ""),
        "user_name": anchor.get("display_name", ""),
        "current_timestamp_utc": timestamp,
        "local_time_context": local_time_context_from_storage_utc(timestamp),
        "prompt_message_context": _prompt_message_context(anchor),
        "channel_topic": "",
        "chat_history_recent": history_recent,
        "chat_history_wide": history_wide,
        "reply_context": anchor.get("reply_context") or {},
        "indirect_speech_context": "",
        "conversation_progress": {
            "status": "active",
            "recent_context_source": context_source,
        },
    }
    return context


def _trimmed_window(
    messages: list[dict[str, Any]],
    *,
    anchor_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Return a production-trimmed sliding history window before the anchor."""

    start = max(0, anchor_index - limit)
    window = trim_history_dict(messages[start:anchor_index])
    return window


def _build_context_modes(
    messages: list[dict[str, Any]],
    *,
    anchor_index: int,
    anchor_message_id: str,
) -> list[dict[str, Any]]:
    """Create current, expanded, and graph-derived context variants."""

    anchor = messages[anchor_index]
    current_recent = _trimmed_window(
        messages,
        anchor_index=anchor_index,
        limit=CURRENT_RECENT_LIMIT,
    )
    current_wide = _trimmed_window(
        messages,
        anchor_index=anchor_index,
        limit=CURRENT_WIDE_LIMIT,
    )
    expanded_window = _trimmed_window(
        messages,
        anchor_index=anchor_index,
        limit=EXPANDED_WINDOW_LIMIT,
    )
    graph_history = _graph_history(
        messages[:anchor_index + 1],
        anchor_message_id=anchor_message_id,
    )
    modes = [
        {
            "mode": "current_trimmed_window",
            "context": _base_runtime_context(
                anchor=anchor,
                history_recent=current_recent,
                history_wide=current_wide,
                context_source="current_trimmed_window",
            ),
        },
        {
            "mode": "expanded_trimmed_window_50",
            "context": _base_runtime_context(
                anchor=anchor,
                history_recent=expanded_window,
                history_wide=expanded_window,
                context_source="expanded_trimmed_window_50",
            ),
        },
        {
            "mode": "conversation_graph_flow",
            "context": _base_runtime_context(
                anchor=anchor,
                history_recent=graph_history,
                history_wide=graph_history,
                context_source="conversation_graph_flow",
            ),
        },
    ]
    return modes


def _slot_counts(slots: list[str]) -> dict[str, int]:
    """Count retrieval slots by top-level initializer prefix."""

    counts = {
        "total": len(slots),
        "Conversation-evidence": 0,
        "Memory-evidence": 0,
        "Person-context": 0,
        "Recall": 0,
        "Live-context": 0,
        "Web-evidence": 0,
    }
    for slot in slots:
        prefix = slot.split(":", maxsplit=1)[0]
        if prefix in counts:
            counts[prefix] += 1
    return counts


async def _run_real_data_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    export_path: Path,
    anchor_message_id: str,
) -> dict[str, Any]:
    """Run one real-data initializer comparison case and write a trace."""

    await _skip_if_llm_unavailable()
    messages = _load_export_messages(export_path)
    anchor_index = _find_message_index(messages, anchor_message_id)
    anchor = messages[anchor_index]

    no_cache = _NoCacheRuntime()
    capturing_llm = _CapturingInitializerLLM(initializer_module._initializer_llm)
    monkeypatch.setattr(initializer_module, "get_rag_cache2_runtime", lambda: no_cache)
    monkeypatch.setattr(initializer_module, "_initializer_llm", capturing_llm)
    monkeypatch.setattr(initializer_module, "_write_initializer_cache", _noop_async)
    monkeypatch.setattr(initializer_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(initializer_module, "record_initializer_hit", _noop_async)

    mode_results: list[dict[str, Any]] = []
    modes = _build_context_modes(
        messages,
        anchor_index=anchor_index,
        anchor_message_id=anchor_message_id,
    )
    for mode in modes:
        before_call_count = len(capturing_llm.calls)
        state = {
            "original_query": str(anchor.get("body_text") or ""),
            "character_name": _character_name(messages),
            "context": mode["context"],
        }
        result = await initializer_module.rag_initializer(state)
        assert len(capturing_llm.calls) == before_call_count + 1
        call_record = capturing_llm.calls[-1]
        unknown_slots = result["unknown_slots"]
        mode_results.append(
            {
                "mode": mode["mode"],
                "context_projected_for_llm": project_runtime_context_for_llm(
                    mode["context"]
                ),
                "raw_prompt_messages": call_record["messages"],
                "raw_model_output": call_record["raw_output"],
                "parsed_result": result,
                "slot_counts": _slot_counts(unknown_slots),
            }
        )

    trace_payload = {
        "case_id": case_id,
        "export_path": str(export_path.relative_to(PROJECT_ROOT)),
        "anchor_message": anchor,
        "anchor_index": anchor_index,
        "model_route": {
            "base_url": RAG_PLANNER_LLM_BASE_URL,
        },
        "context_modes": mode_results,
        "judgment": "manual_review_required_for_real_data_initializer_load",
    }
    trace_path = write_llm_trace(
        "rag_initializer_real_data_live_llm",
        case_id,
        trace_payload,
    )
    logger.info(f"RAG_INITIALIZER_REAL_DATA case={case_id} trace={trace_path}")

    return trace_payload


async def test_real_data_group_attachment_followup_sam_purchase_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect group attachment follow-up where flat history loses image text."""

    payload = await _run_real_data_case(
        monkeypatch,
        case_id="group_attachment_followup_sam_purchase_power",
        export_path=GROUP_EXPORT_PATH,
        anchor_message_id="1698008124",
    )
    assert payload["context_modes"]


async def test_real_data_group_parallel_attachment_comment_sam_free_food(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect group parallel-thread attachment comment with nearby noise."""

    payload = await _run_real_data_case(
        monkeypatch,
        case_id="group_parallel_attachment_comment_sam_free_food",
        export_path=GROUP_EXPORT_PATH,
        anchor_message_id="1798628230",
    )
    assert payload["context_modes"]


async def test_real_data_private_active_homework_continuity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect private active-agreement recall pressure from real chat."""

    payload = await _run_real_data_case(
        monkeypatch,
        case_id="private_active_homework_continuity",
        export_path=PRIVATE_EXPORT_PATH,
        anchor_message_id="889651108",
    )
    assert payload["context_modes"]


async def test_real_data_private_character_preference_local_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect local character-preference question grounded by reply context."""

    payload = await _run_real_data_case(
        monkeypatch,
        case_id="private_character_preference_local_reply",
        export_path=PRIVATE_EXPORT_PATH,
        anchor_message_id="1662583774",
    )
    assert payload["context_modes"]


async def test_real_data_private_recent_reply_no_search_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect a local private reply that should not need RAG search."""

    payload = await _run_real_data_case(
        monkeypatch,
        case_id="private_recent_reply_no_search_control",
        export_path=PRIVATE_EXPORT_PATH,
        anchor_message_id="822357735",
    )
    assert payload["context_modes"]
