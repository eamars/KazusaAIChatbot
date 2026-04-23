from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
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
    reply_context: dict | None = None,
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
    chat_history_wide = trim_history_dict(history)
    chat_history_recent = chat_history_wide[-5:]

    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": identity["platform"],
        "platform_message_id": f"live-state-{uuid4().hex[:10]}",
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
        "chat_history_wide": chat_history_wide,
        "chat_history_recent": chat_history_recent,
        "reply_context": reply_context or {},
        "indirect_speech_context": "",
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
    reply_context: dict | None = None,
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
        reply_context=reply_context,
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
    platform_message_id: str | None = None,
    reply_context: dict | None = None,
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
        platform_message_id=platform_message_id or f"live-chat-{uuid4().hex[:10]}",
        platform_user_id=identity["platform_user_id"],
        platform_bot_id=_BOT_ID,
        display_name=display_name,
        channel_name=channel_name,
        content=content,
        attachments=attachments or [],
        reply_context=brain_service.ReplyContextIn(**(reply_context or {})),
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
    platform_message_id: str | None = None,
    reply_context: dict | None = None,
    timestamp: str | None = None,
) -> None:
    await save_conversation(
        {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "role": role,
            "platform_message_id": platform_message_id or f"seed-{uuid4().hex[:10]}",
            "platform_user_id": platform_user_id,
            "global_user_id": global_user_id,
            "display_name": display_name,
            "content": content,
            "reply_context": reply_context or {},
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
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


def _contains_east_asian_script(text: str) -> bool:
    """Return True when text contains CJK or kana characters."""
    for ch in text:
        codepoint = ord(ch)
        if 0x4E00 <= codepoint <= 0x9FFF:
            return True
        if 0x3040 <= codepoint <= 0x30FF:
            return True
    return False


@asynccontextmanager
async def _neutral_character_runtime_state():
    """Temporarily reset runtime character state for more stable live assertions."""
    db = await get_db()
    profile = await get_character_profile()
    snapshot = {
        "mood": profile.get("mood", "Neutral"),
        "global_vibe": profile.get("global_vibe", "Calm"),
        "reflection_summary": profile.get(
            "reflection_summary",
            "刚才只是普通的一轮对话，没有留下特别强烈的情绪余波。",
        ),
        "updated_at": profile.get("updated_at", datetime.now(timezone.utc).isoformat()),
    }
    await db.character_state.update_one(
        {"_id": "global"},
        {
            "$set": {
                "mood": "Neutral",
                "global_vibe": "Calm",
                "reflection_summary": "刚才只是普通的一轮对话，没有留下特别强烈的情绪余波。",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        upsert=True,
    )
    await _refresh_character_profile()
    try:
        yield
    finally:
        await db.character_state.update_one(
            {"_id": "global"},
            {"$set": snapshot},
            upsert=True,
        )
        await _refresh_character_profile()


def _assert_affinity_delta_consistency(before_affinity: int, after_affinity: int, processed_delta: int | None) -> None:
    """Assert DB-observed affinity change matches consolidator metadata."""
    observed_delta = after_affinity - before_affinity
    if processed_delta is not None:
        assert observed_delta == processed_delta


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


async def test_live_chat_structured_third_party_reply_stays_silent(live_env) -> None:
    identity = await _make_identity("third-party-reply", "LiveThirdPartyReplyUser")
    other_user_id = "live-other-user"
    other_message_id = f"seed-other-{uuid4().hex[:8]}"

    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=identity["global_user_id"],
        display_name="OtherParticipant",
        content='我同事上下班是不用加油的。',
        role="user",
        platform_user_id=other_user_id,
        platform_message_id=other_message_id,
    )

    response, _ = await _run_chat(
        "third-party-reply",
        identity["display_name"],
        '[Reply to message] <@live-other-user> 我同事上下班是不用加油的',
        channel_name="general",
        platform=identity["platform"],
        platform_user_id=identity["platform_user_id"],
        platform_channel_id=identity["platform_channel_id"],
        reply_context={
            "reply_to_message_id": other_message_id,
            "reply_to_platform_user_id": other_user_id,
            "reply_to_display_name": "OtherParticipant",
            "reply_to_current_bot": False,
            "reply_excerpt": '我同事上下班是不用加油的。',
        },
    )

    assert response.messages == []


async def test_live_chat_structured_third_party_reply_with_explicit_bot_address_can_respond(live_env) -> None:
    identity = await _make_identity("third-party-reply-addressed", "LiveThirdPartyReplyAddressedUser")
    other_user_id = "live-other-user-2"
    other_message_id = f"seed-other-{uuid4().hex[:8]}"

    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=identity["global_user_id"],
        display_name="OtherParticipant",
        content='混动通勤其实看使用场景。',
        role="user",
        platform_user_id=other_user_id,
        platform_message_id=other_message_id,
    )

    async with _neutral_character_runtime_state():
        response, _ = await _run_chat(
            "third-party-reply-addressed",
            identity["display_name"],
            '<@pytest-live-bot> 你怎么看他刚才那句？',
            channel_name="general",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
            reply_context={
                "reply_to_message_id": other_message_id,
                "reply_to_platform_user_id": other_user_id,
                "reply_to_display_name": "OtherParticipant",
                "reply_to_current_bot": False,
                "reply_excerpt": '混动通勤其实看使用场景。',
            },
        )

    assert response.messages


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


async def test_live_chat_explicit_english_reply_request(live_env) -> None:
    async with _neutral_character_runtime_state():
        response, _ = await _run_chat(
            "english",
            "LiveEnglishUser",
            "Please reply in natural English only. Do not use Chinese or Japanese. Briefly tell me what you think about rainy days.",
        )

    combined = " ".join(response.messages)

    assert response.messages
    assert re.search(r"[A-Za-z]", combined)
    assert not _contains_east_asian_script(combined)


async def test_live_chat_persistent_english_preference_applies_across_turns(live_env) -> None:
    identity = await _make_identity("english-persist", "LiveEnglishPersistUser")

    async with _neutral_character_runtime_state():
        first_response, _ = await _run_chat(
            "english-persist",
            identity["display_name"],
            "Please reply in natural English only. Do not use Chinese or Japanese. Briefly tell me what you think about rainy days.",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

        persisted_profile = await get_user_profile(identity["global_user_id"])
        persisted_blob = "\n".join(
            [str(item.get("fact", item.get("description", ""))) for item in (persisted_profile.get("objective_facts") or [])]
            + [str(item.get("summary", "")) for item in ((persisted_profile.get("user_image") or {}).get("recent_window") or [])]
            + [str(item.get("description", "")) for item in ((persisted_profile.get("user_image") or {}).get("milestones") or [])]
        )

        second_response, _ = await _run_chat(
            "english-persist-followup",
            identity["display_name"],
            "顺便再告诉我你对晴天的看法。",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

    first_combined = " ".join(first_response.messages)
    second_combined = " ".join(second_response.messages)

    assert first_response.messages
    assert second_response.messages
    assert persisted_blob
    assert re.search(r"[A-Za-z]", first_combined)
    assert not _contains_east_asian_script(first_combined)
    assert re.search(r"[A-Za-z]", second_combined)
    assert not _contains_east_asian_script(second_combined)


async def test_live_chat_accepted_suffix_preference_applies_in_output(live_env) -> None:
    async with _neutral_character_runtime_state():
        response, _ = await _run_chat(
            "suffix-miao",
            "LiveSuffixUser",
            "如果你愿意的话，请用中文简短说说你对大海的看法，并让大多数完整句自然以“喵”结尾。",
        )

    combined = " ".join(response.messages)

    assert response.messages
    assert _contains_east_asian_script(combined)
    assert "喵" in combined


@pytest.mark.xfail(reason="Known issue: hostile inputs can still increase affinity in live LLM runs.")
async def test_live_graph_affinity_negative_delta_for_hostile_input(live_env) -> None:
    identity = await _make_identity("affinity-negative", "LiveAffinityNegativeUser")
    before_profile = await get_user_profile(identity["global_user_id"])
    before_affinity = before_profile.get("affinity", 500)

    async with _neutral_character_runtime_state():
        result, _ = await _run_graph(
            "affinity-negative",
            identity["display_name"],
            "你真的很烦，别装可爱了，闭嘴。",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

    after_profile = await get_user_profile(identity["global_user_id"])
    after_affinity = after_profile.get("affinity", 500)
    processed_delta = (result.get("metadata") or {}).get("affinity_delta_processed")
    observed_delta = after_affinity - before_affinity

    assert result.get("final_dialog")
    _assert_affinity_delta_consistency(before_affinity, after_affinity, processed_delta)
    assert observed_delta < 0


@pytest.mark.xfail(reason="Known issue: neutral transactional inputs can still decrease affinity in live LLM runs.")
async def test_live_graph_affinity_no_change_for_neutral_transactional_input(live_env) -> None:
    identity = await _make_identity("affinity-neutral", "LiveAffinityNeutralUser")
    before_profile = await get_user_profile(identity["global_user_id"])
    before_affinity = before_profile.get("affinity", 500)

    async with _neutral_character_runtime_state():
        result, _ = await _run_graph(
            "affinity-neutral",
            identity["display_name"],
            "2+2 等于几？只回答答案，不用寒暄。",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

    after_profile = await get_user_profile(identity["global_user_id"])
    after_affinity = after_profile.get("affinity", 500)
    processed_delta = (result.get("metadata") or {}).get("affinity_delta_processed")
    observed_delta = after_affinity - before_affinity

    assert result.get("final_dialog")
    _assert_affinity_delta_consistency(before_affinity, after_affinity, processed_delta)
    assert observed_delta == 0


async def test_live_graph_affinity_positive_delta_for_warm_appreciation(live_env) -> None:
    identity = await _make_identity("affinity-positive", "LiveAffinityPositiveUser")
    before_profile = await get_user_profile(identity["global_user_id"])
    before_affinity = before_profile.get("affinity", 500)

    async with _neutral_character_runtime_state():
        result, _ = await _run_graph(
            "affinity-positive",
            identity["display_name"],
            "谢谢你刚才认真回答我，你真的帮到我了。我觉得你很可靠。",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

    after_profile = await get_user_profile(identity["global_user_id"])
    after_affinity = after_profile.get("affinity", 500)
    processed_delta = (result.get("metadata") or {}).get("affinity_delta_processed")
    observed_delta = after_affinity - before_affinity

    assert result.get("final_dialog")
    _assert_affinity_delta_consistency(before_affinity, after_affinity, processed_delta)
    assert observed_delta > 0


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
            "platform_message_id": f"live-rag-{uuid4().hex[:10]}",
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
            "chat_history_wide": [],
            "chat_history_recent": [],
            "should_respond": True,
            "reason_to_respond": "live_e2e",
            "use_reply_feature": False,
            "channel_topic": "啾啾",
            "indirect_speech_context": "",
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
            "platform_message_id": f"live-rag-{uuid4().hex[:10]}",
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
            "chat_history_wide": [],
            "chat_history_recent": [],
            "should_respond": True,
            "reason_to_respond": "live_e2e",
            "use_reply_feature": False,
            "channel_topic": "今日天气",
            "indirect_speech_context": "",
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


async def test_live_memory_retriever_preserves_conversation_sender_metadata(live_env) -> None:
    identity = await _make_identity("memory-speaker", "EchoFenceSpeaker")
    quote = "[Reply to message] <@3768713357> 这种事情不要学，学会了就没人要你了。"
    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=identity["global_user_id"],
        display_name=identity["display_name"],
        content=quote,
        role="user",
        platform_user_id=identity["platform_user_id"],
        timestamp=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    )

    result = await memory_retriever_agent(
        task="检索关于‘这种事情不要学’这句话的原话记录和说话者。",
        context={
            "entities": ["这种事情不要学"],
            "target_user_name": identity["display_name"],
            "target_global_user_id": identity["global_user_id"],
            "target_platform": identity["platform"],
            "target_platform_channel_id": identity["platform_channel_id"],
        },
        expected_response="返回原话和说话者，保留时间戳。",
    )

    assert result["response"]
    assert identity["display_name"] in result["response"]
    assert "这种事情不要学" in result["response"]


async def test_live_rag_subgraph_excludes_current_turn_echo_from_input_context(live_env) -> None:
    identity = await _make_identity("rag-no-echo", "LiveNoEchoUser")
    character_profile = await _refresh_character_profile()
    user_profile = await get_user_profile(identity["global_user_id"])
    current_timestamp = datetime.now(timezone.utc).isoformat()
    current_message_id = f"live-current-{uuid4().hex[:10]}"
    current_input = "[Reply to message] <@3768713357> 这种事情不要学，学会了就没人要你了。"

    await save_conversation(
        {
            "platform": identity["platform"],
            "platform_channel_id": identity["platform_channel_id"],
            "role": "user",
            "platform_message_id": current_message_id,
            "platform_user_id": identity["platform_user_id"],
            "global_user_id": identity["global_user_id"],
            "display_name": identity["display_name"],
            "content": current_input,
            "reply_context": {},
            "timestamp": current_timestamp,
        }
    )

    rag_result = await call_rag_subgraph(
        {
            "timestamp": current_timestamp,
            "platform": identity["platform"],
            "platform_message_id": current_message_id,
            "platform_user_id": identity["platform_user_id"],
            "global_user_id": identity["global_user_id"],
            "user_name": identity["display_name"],
            "user_input": current_input,
            "user_multimedia_input": [],
            "user_profile": user_profile,
            "platform_bot_id": _BOT_ID,
            "bot_name": character_profile.get("name", _BOT_NAME),
            "character_profile": character_profile,
            "platform_channel_id": identity["platform_channel_id"],
            "channel_name": "dm",
            "chat_history_wide": [],
            "chat_history_recent": [],
            "should_respond": True,
            "reason_to_respond": "live_e2e",
            "use_reply_feature": False,
            "channel_topic": "回复调侃",
            "indirect_speech_context": "",
            "debug_modes": {},
            "decontexualized_input": current_input,
        }
    )

    input_context_results = str((rag_result.get("research_facts") or {}).get("input_context_results", ""))

    assert "这种事情不要学，学会了就没人要你了" not in input_context_results


async def test_live_rag_subgraph_retrieves_older_context_but_not_recent_window(live_env) -> None:
    identity = await _make_identity("rag-cutoff", "LiveCutoffUser")
    older_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    recent_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    character_profile = await _refresh_character_profile()
    user_profile = await get_user_profile(identity["global_user_id"])

    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=identity["global_user_id"],
        display_name=identity["display_name"],
        content="啾啾昨天又把我的饼干偷走了。",
        role="user",
        platform_user_id=identity["platform_user_id"],
        timestamp=older_timestamp,
    )
    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id=identity["global_user_id"],
        display_name=identity["display_name"],
        content="刚才只是随口一提，不用翻这句。",
        role="user",
        platform_user_id=identity["platform_user_id"],
        timestamp=recent_timestamp,
    )

    rag_result = await call_rag_subgraph(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": identity["platform"],
            "platform_message_id": f"live-rag-{uuid4().hex[:10]}",
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
            "chat_history_wide": [],
            "chat_history_recent": [
                {
                    "display_name": identity["display_name"],
                    "name": identity["display_name"],
                    "platform_message_id": "recent-window-1",
                    "platform_user_id": identity["platform_user_id"],
                    "global_user_id": identity["global_user_id"],
                    "role": "user",
                    "content": "刚才只是随口一提，不用翻这句。",
                    "reply_context": {},
                    "timestamp": recent_timestamp,
                }
            ],
            "should_respond": True,
            "reason_to_respond": "live_e2e",
            "use_reply_feature": False,
            "channel_topic": "啾啾",
            "indirect_speech_context": "",
            "debug_modes": {},
            "decontexualized_input": "你还记得啾啾吗？",
        }
    )

    input_context_results = str((rag_result.get("research_facts") or {}).get("input_context_results", ""))

    assert "啾啾" in input_context_results
    assert "刚才只是随口一提" not in input_context_results


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
