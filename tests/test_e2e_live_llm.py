from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.agents.memory_retriever_agent import memory_retriever_agent
from kazusa_ai_chatbot.agents.web_search_agent2 import web_search_agent
from kazusa_ai_chatbot.config import LLM_BASE_URL, SCHEDULED_TASKS_ENABLED
from kazusa_ai_chatbot.db import (
    build_memory_doc,
    close_db,
    db_bootstrap,
    get_character_profile,
    get_conversation_history,
    get_db,
    get_user_profile,
    resolve_global_user_id,
    save_conversation,
    save_memory,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag as rag_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache, call_rag_subgraph
from kazusa_ai_chatbot.utils import trim_history_dict

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_IMAGE_PATH = Path(__file__).resolve().parents[1] / "personalities" / "kazusa.png"
_BOT_ID = "pytest-live-bot"
_BOT_NAME = "KazusaLiveBot"


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {LLM_BASE_URL}")


@pytest_asyncio.fixture()
async def live_env():
    await _skip_if_llm_unavailable()
    await db_bootstrap()
    character_profile = await get_character_profile()
    if not character_profile.get("name"):
        pytest.fail("Character profile is missing from MongoDB.")

    brain_service._personality = character_profile
    brain_service._graph = brain_service._build_graph()
    brain_service._chat_executor_semaphore = None

    await mcp_manager.start()
    await _get_rag_cache()

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.load_pending_events()

    yield {
        "mcp_tools": {tool.name for tool in mcp_manager.list_tools()},
    }

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.shutdown()

    cache = await _get_rag_cache()
    await cache.shutdown()
    rag_module._rag_cache = None
    await mcp_manager.stop()
    await close_db()


async def _refresh_character_profile() -> dict:
    character_profile = await get_character_profile()
    brain_service._personality = character_profile
    return character_profile


async def _make_identity(label: str, display_name: str) -> dict:
    suffix = uuid4().hex[:10]
    platform = f"pytest-live-{label}"
    platform_user_id = f"user-{suffix}"
    platform_channel_id = f"channel-{suffix}"
    global_user_id = await resolve_global_user_id(
        platform=platform,
        platform_user_id=platform_user_id,
        display_name=display_name,
    )
    return {
        "platform": platform,
        "platform_user_id": platform_user_id,
        "platform_channel_id": platform_channel_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
    }


async def _make_initial_state(
    label: str,
    display_name: str,
    content: str,
    *,
    channel_name: str = "dm",
    user_multimedia_input: list[dict] | None = None,
    platform: str | None = None,
    platform_user_id: str | None = None,
    platform_channel_id: str | None = None,
) -> tuple[dict, dict]:
    if platform is None or platform_user_id is None or platform_channel_id is None:
        identity = await _make_identity(label, display_name)
    else:
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=display_name,
        )
        identity = {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "platform_channel_id": platform_channel_id,
            "global_user_id": global_user_id,
            "display_name": display_name,
        }

    character_profile = await _refresh_character_profile()
    user_profile = await get_user_profile(identity["global_user_id"])
    history = await get_conversation_history(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
    )

    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": identity["platform"],
        "platform_user_id": identity["platform_user_id"],
        "global_user_id": identity["global_user_id"],
        "user_name": identity["display_name"],
        "user_input": content,
        "user_multimedia_input": user_multimedia_input or [],
        "user_profile": user_profile,
        "platform_bot_id": _BOT_ID,
        "bot_name": character_profile.get("name", _BOT_NAME),
        "character_profile": character_profile,
        "platform_channel_id": identity["platform_channel_id"],
        "channel_name": channel_name,
        "chat_history": trim_history_dict(history),
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
    }
    return state, identity


async def _run_graph(
    label: str,
    display_name: str,
    content: str,
    *,
    channel_name: str = "dm",
    user_multimedia_input: list[dict] | None = None,
    platform: str | None = None,
    platform_user_id: str | None = None,
    platform_channel_id: str | None = None,
) -> tuple[dict, dict]:
    state, identity = await _make_initial_state(
        label,
        display_name,
        content,
        channel_name=channel_name,
        user_multimedia_input=user_multimedia_input,
        platform=platform,
        platform_user_id=platform_user_id,
        platform_channel_id=platform_channel_id,
    )
    result = await brain_service._graph.ainvoke(state)
    return result, identity


async def _run_chat(
    label: str,
    display_name: str,
    content: str,
    *,
    channel_name: str = "dm",
    attachments: list[brain_service.AttachmentIn] | None = None,
    platform: str | None = None,
    platform_user_id: str | None = None,
    platform_channel_id: str | None = None,
) -> tuple[brain_service.ChatResponse, dict]:
    if platform is None or platform_user_id is None or platform_channel_id is None:
        identity = await _make_identity(label, display_name)
    else:
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=display_name,
        )
        identity = {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "platform_channel_id": platform_channel_id,
            "global_user_id": global_user_id,
            "display_name": display_name,
        }

    await _refresh_character_profile()
    background_tasks = BackgroundTasks()
    request = brain_service.ChatRequest(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        platform_user_id=identity["platform_user_id"],
        platform_bot_id=_BOT_ID,
        display_name=display_name,
        channel_name=channel_name,
        content=content,
        attachments=attachments or [],
    )
    response = await brain_service.chat(request, background_tasks)
    for task in background_tasks.tasks:
        await task()
    return response, identity


async def _seed_conversation(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    display_name: str,
    content: str,
    role: str,
    platform_user_id: str,
) -> None:
    await save_conversation(
        {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "role": role,
            "platform_user_id": platform_user_id,
            "global_user_id": global_user_id,
            "display_name": display_name,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


async def _seed_memory(global_user_id: str, memory_name: str, content: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    doc = build_memory_doc(
        memory_name=memory_name,
        content=content,
        source_global_user_id=global_user_id,
        memory_type="fact",
        source_kind="conversation_extracted",
        confidence_note="live_e2e_seed",
    )
    await save_memory(doc, timestamp)


def _load_test_image_b64() -> str:
    return base64.b64encode(_IMAGE_PATH.read_bytes()).decode("utf-8")


async def test_live_chat_smoke_response(live_env) -> None:
    response, _ = await _run_chat(
        "smoke",
        "LiveSmokeUser",
        "千纱，你今天过得怎么样？",
    )

    assert response.messages
    assert response.content_type == "text"


async def test_live_chat_third_party_mention_stays_silent(live_env) -> None:
    response, _ = await _run_chat(
        "third-party",
        "LiveThirdPartyUser",
        "<@someone-else> 你去问他吧，别问我。",
        channel_name="general",
    )

    assert response.messages == []


async def test_live_chat_multimodal_image_response(live_env) -> None:
    response, _ = await _run_chat(
        "image",
        "LiveImageUser",
        "看看这张图里有什么。",
        attachments=[
            brain_service.AttachmentIn(
                media_type="image/png",
                base64_data=_load_test_image_b64(),
            )
        ],
    )

    assert response.messages


async def test_live_graph_fact_extraction_persists_profile_updates(live_env) -> None:
    identity = await _make_identity("facts", "LiveFactsUser")
    before_profile = await get_user_profile(identity["global_user_id"])
    before_facts = list(before_profile.get("objective_facts") or [])
    before_recent = list((before_profile.get("user_image") or {}).get("recent_window") or [])
    before_milestones = list((before_profile.get("user_image") or {}).get("milestones") or [])

    result, _ = await _run_graph(
        "facts",
        identity["display_name"],
        "我要告诉你一件很重要的事：我一直住在奥克兰，而且我永远不吃辣椒。我现在在新西兰做软件工程师。",
        platform=identity["platform"],
        platform_user_id=identity["platform_user_id"],
        platform_channel_id=identity["platform_channel_id"],
    )

    after_profile = await get_user_profile(identity["global_user_id"])
    after_facts = list(after_profile.get("objective_facts") or [])
    after_recent = list((after_profile.get("user_image") or {}).get("recent_window") or [])
    after_milestones = list((after_profile.get("user_image") or {}).get("milestones") or [])
    persisted_blob = "\n".join(
        [str(item.get("fact", "")) for item in after_facts]
        + [str(item.get("summary", "")) for item in after_recent]
        + [str(item.get("event", item.get("description", ""))) for item in after_milestones]
    )

    assert result.get("final_dialog")
    assert (
        len(after_facts) > len(before_facts)
        or len(after_recent) > len(before_recent)
        or len(after_milestones) > len(before_milestones)
    )
    assert any(keyword in persisted_blob for keyword in ("奥克兰", "软件工程师", "辣椒"))


async def test_live_graph_future_promise_creates_scheduled_event(live_env) -> None:
    if not SCHEDULED_TASKS_ENABLED:
        pytest.skip("Scheduled tasks are disabled in this environment.")

    identity = await _make_identity("promise", "LivePromiseUser")
    db = await get_db()
    before_count = await db.scheduled_events.count_documents(
        {
            "target_global_user_id": identity["global_user_id"],
            "event_type": "future_promise",
        }
    )

    result, _ = await _run_graph(
        "promise",
        identity["display_name"],
        "如果我今晚把作业写完，你明天早上要奖励我，别忘了。",
        platform=identity["platform"],
        platform_user_id=identity["platform_user_id"],
        platform_channel_id=identity["platform_channel_id"],
    )

    after_count = await db.scheduled_events.count_documents(
        {
            "target_global_user_id": identity["global_user_id"],
            "event_type": "future_promise",
        }
    )

    assert result.get("final_dialog")
    assert result.get("future_promises")
    assert after_count > before_count


async def test_live_rag_subgraph_retrieves_seeded_context(live_env) -> None:
    identity = await _make_identity("rag-memory", "LiveRagUser")
    await _seed_memory(
        identity["global_user_id"],
        "[啾啾] profile",
        "啾啾是一只总爱抢零食的黑猫，最近一次被提到是在上周。",
    )
    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=identity["global_user_id"],
        display_name=identity["display_name"],
        content="啾啾昨天又把我的饼干偷走了。",
        role="user",
        platform_user_id=identity["platform_user_id"],
    )

    character_profile = await _refresh_character_profile()
    user_profile = await get_user_profile(identity["global_user_id"])
    rag_result = await call_rag_subgraph(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": identity["platform"],
            "platform_user_id": identity["platform_user_id"],
            "global_user_id": identity["global_user_id"],
            "user_name": identity["display_name"],
            "user_input": "你还记得啾啾吗？",
            "user_multimedia_input": [],
            "user_profile": user_profile,
            "platform_bot_id": _BOT_ID,
            "bot_name": character_profile.get("name", _BOT_NAME),
            "character_profile": character_profile,
            "platform_channel_id": identity["platform_channel_id"],
            "channel_name": "dm",
            "chat_history": [],
            "should_respond": True,
            "reason_to_respond": "live_e2e",
            "use_reply_feature": False,
            "channel_topic": "啾啾",
            "user_topic": "询问关于啾啾的记忆",
            "debug_modes": {},
            "decontexualized_input": "你还记得啾啾吗？",
        }
    )

    input_context_results = str((rag_result.get("research_facts") or {}).get("input_context_results", ""))

    assert input_context_results
    assert "啾啾" in input_context_results


async def test_live_rag_subgraph_dispatches_external_search(live_env) -> None:
    required_tools = {
        "mcp-searxng__searxng_web_search",
        "mcp-searxng__web_url_read",
    }
    if not required_tools.issubset(live_env["mcp_tools"]):
        pytest.skip("SearXNG MCP tools are not configured in this environment.")

    identity = await _make_identity("rag-web", "LiveRagWebUser")
    character_profile = await _refresh_character_profile()
    user_profile = await get_user_profile(identity["global_user_id"])
    rag_result = await call_rag_subgraph(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": identity["platform"],
            "platform_user_id": identity["platform_user_id"],
            "global_user_id": identity["global_user_id"],
            "user_name": identity["display_name"],
            "user_input": "查一下今天奥克兰的天气。",
            "user_multimedia_input": [],
            "user_profile": user_profile,
            "platform_bot_id": _BOT_ID,
            "bot_name": character_profile.get("name", _BOT_NAME),
            "character_profile": character_profile,
            "platform_channel_id": identity["platform_channel_id"],
            "channel_name": "dm",
            "chat_history": [],
            "should_respond": True,
            "reason_to_respond": "live_e2e",
            "use_reply_feature": False,
            "channel_topic": "今日天气",
            "user_topic": "查询今天奥克兰的实时天气",
            "debug_modes": {},
            "decontexualized_input": "查一下今天奥克兰的天气。",
        }
    )

    external_rag_results = str((rag_result.get("research_facts") or {}).get("external_rag_results", ""))

    assert external_rag_results
    assert "奥克兰" in external_rag_results or "Auckland" in external_rag_results


async def test_live_memory_retriever_agent_reads_seeded_memory(live_env) -> None:
    identity = await _make_identity("memory-agent", "LiveMemoryAgentUser")
    await _seed_memory(
        identity["global_user_id"],
        "[啾啾] notes",
        "啾啾是经常偷点心的黑猫，喜欢在桌子旁边打转。",
    )

    result = await memory_retriever_agent(
        task="检索关于啾啾的身份描述和行为特征。",
        context={
            "entities": ["啾啾"],
            "target_user_name": identity["display_name"],
            "target_global_user_id": identity["global_user_id"],
        },
        expected_response="用中文说明啾啾是谁，以及有什么典型行为。",
    )

    assert result["status"] in {"complete", "partial", "incomplete"}
    assert result["response"]
    assert "啾啾" in result["response"]


async def test_live_web_search_agent_returns_live_result(live_env) -> None:
    required_tools = {
        "mcp-searxng__searxng_web_search",
        "mcp-searxng__web_url_read",
    }
    if not required_tools.issubset(live_env["mcp_tools"]):
        pytest.skip("SearXNG MCP tools are not configured in this environment.")

    result = await web_search_agent(
        task="查找今天奥克兰的天气，并用中文简短说明当前情况，附一个来源链接。",
        context={},
        expected_response="中文短答，包含 1 个来源链接。",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    assert result["status"] in {"success", "partial", "not_found"}
    assert result["response"]
