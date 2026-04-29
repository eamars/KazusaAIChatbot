from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import re
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.config import DIALOG_GENERATOR_LLM_BASE_URL, SCHEDULED_TASKS_ENABLED
from kazusa_ai_chatbot.db import (
    build_memory_doc,
    close_db,
    db_bootstrap,
    get_character_profile,
    get_conversation_history,
    get_db,
    get_user_profile,
    insert_user_memory_units,
    query_user_memory_units,
    resolve_global_user_id,
    save_conversation,
    save_memory,
    UserMemoryUnitType,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.rag.web_search_agent import _run_subgraph as web_search_agent
from kazusa_ai_chatbot.utils import trim_history_dict
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_IMAGE_PATH = Path(__file__).resolve().parents[1] / "personalities" / "kazusa.png"
_BOT_ID = "pytest-live-bot"
_BOT_NAME = "KazusaLiveBot"
logger = logging.getLogger(__name__)


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{DIALOG_GENERATOR_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {DIALOG_GENERATOR_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {DIALOG_GENERATOR_LLM_BASE_URL}")


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

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.load_pending_events()

    yield {
        "mcp_tools": {tool.name for tool in mcp_manager.list_tools()},
    }

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.shutdown()

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
    write_llm_trace(
        "e2e_live_graph",
        label,
        {
            "input": state,
            "result": result,
            "judgment": "graph_output_matches_case_assertions_when_test_passes",
        },
    )
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
    write_llm_trace(
        "e2e_live_chat",
        label,
        {
            "request": request.model_dump(),
            "response": response.model_dump(),
            "judgment": "chat_response_matches_case_assertions_when_test_passes",
        },
    )
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


async def _make_group_identities(label: str, display_names: list[str]) -> dict[str, dict]:
    shared_suffix = uuid4().hex[:10]
    platform = f"pytest-live-{label}"
    platform_channel_id = f"channel-{shared_suffix}"
    identities = {}
    for index, display_name in enumerate(display_names):
        platform_user_id = f"user-{index}-{uuid4().hex[:8]}"
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=display_name,
        )
        identities[display_name] = {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "platform_channel_id": platform_channel_id,
            "global_user_id": global_user_id,
            "display_name": display_name,
        }
    return identities


async def _seed_group_series(identities: dict[str, dict], messages: list[dict], bot_display_name: str) -> None:
    base_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    reference_identity = next(iter(identities.values()))
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            identity = {
                "platform": reference_identity["platform"],
                "platform_channel_id": reference_identity["platform_channel_id"],
                "global_user_id": "",
                "display_name": bot_display_name,
                "platform_user_id": _BOT_ID,
            }
        else:
            identity = identities[message["speaker"]]
        await _seed_conversation(
            platform=identity["platform"],
            platform_channel_id=identity["platform_channel_id"],
            global_user_id=identity["global_user_id"],
            display_name=identity["display_name"],
            content=content,
            role=role,
            platform_user_id=identity["platform_user_id"],
            timestamp=(base_time + timedelta(seconds=index)).isoformat(),
        )


async def _persist_bot_dialog(identity: dict, bot_display_name: str, dialog: list[str]) -> None:
    if not dialog:
        return
    await _seed_conversation(
        platform=identity["platform"],
        platform_channel_id=identity["platform_channel_id"],
        global_user_id="",
        display_name=bot_display_name,
        content="\n".join(dialog),
        role="assistant",
        platform_user_id=_BOT_ID,
    )


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
    runtime = get_rag_cache2_runtime()
    before_stats = runtime.get_stats()

    response, _ = await _run_chat(
        "smoke",
        "LiveSmokeUser",
        "千纱，你今天过得怎么样？",
    )
    after_stats = runtime.get_stats()

    assert response.messages
    assert response.content_type == "text"
    assert after_stats["hits"] >= before_stats["hits"]
    assert after_stats["misses"] >= before_stats["misses"]
    assert after_stats["size"] <= after_stats["max_entries"]
    logger.info("Cache2 stats before chat: %s", before_stats)
    logger.info("Cache2 stats after chat: %s", after_stats)


async def test_live_chat_multi_user_photo_thread_keeps_user_intents_separated(live_env) -> None:
    character_profile = await _refresh_character_profile()
    bot_display_name = character_profile.get("name", _BOT_NAME)
    identities = await _make_group_identities(
        "photo-thread",
        ["蚝爹油", "猎明"],
    )
    await _seed_group_series(
        identities,
        [
            {"role": "user", "speaker": "蚝爹油", "content": f"<@{_BOT_ID}> 这次能看到我说话了么"},
            {"role": "assistant", "content": "诶？\n你的消息我一直都在看啊。\n能看到，也听得到……\n这种感觉，大概有点不知所措呢。"},
            {"role": "user", "speaker": "蚝爹油", "content": f"[Reply to message] <@{_BOT_ID}> 这是千纱你的照片"},
            {"role": "assistant", "content": "诶……\n这种照片你也看得下去啊？\n明明就是想看我出糗吧，学长。"},
            {"role": "user", "speaker": "猎明", "content": f"<@{_BOT_ID}> [Face] 你照片真涩情"},
        ],
        bot_display_name,
    )

    async with _neutral_character_runtime_state():
        first_prompt = f"<@{_BOT_ID}> 那我前面那句和他夸你照片那个意思，其实不是一回事吧？"
        first_result, _ = await _run_graph(
            "photo-thread-haodieyou",
            identities["蚝爹油"]["display_name"],
            first_prompt,
            channel_name="general",
            platform=identities["蚝爹油"]["platform"],
            platform_user_id=identities["蚝爹油"]["platform_user_id"],
            platform_channel_id=identities["蚝爹油"]["platform_channel_id"],
        )

        first_dialog = first_result.get("final_dialog", [])
        assert first_dialog
        assert "busy right now" not in "\n".join(first_dialog)
        await _persist_bot_dialog(identities["蚝爹油"], bot_display_name, first_dialog)

        second_result, _ = await _run_graph(
            "photo-thread-lieming",
            identities["猎明"]["display_name"],
            f'<@{_BOT_ID}> 那我刚才是在接照片那条，不是在替他发言，对吧？',
            channel_name="general",
            platform=identities["猎明"]["platform"],
            platform_user_id=identities["猎明"]["platform_user_id"],
            platform_channel_id=identities["猎明"]["platform_channel_id"],
        )

    second_dialog = second_result.get("final_dialog", [])
    assert second_dialog
    assert "busy right now" not in "\n".join(second_dialog)
    assert "\n".join(second_dialog) != "\n".join(first_dialog)
    assert "学长" not in "\n".join(second_dialog)


async def test_live_chat_multi_user_quantization_thread_keeps_xuezhang_bound_to_haodieyou(live_env) -> None:
    character_profile = await _refresh_character_profile()
    bot_display_name = character_profile.get("name", _BOT_NAME)
    identities = await _make_group_identities(
        "quantization-thread",
        ["蚝爹油", "C"],
    )
    await _seed_group_series(
        identities,
        [
            {"role": "user", "speaker": "C", "content": "好想要快网速"},
            {"role": "user", "speaker": "C", "content": "好痛苦】"},
            {"role": "user", "speaker": "蚝爹油", "content": f"[Reply to message] <@{_BOT_ID}> 千纱怎么知道量化这种词的？"},
            {"role": "assistant", "content": "诶？刚才那个词……\n学长怎么突然问起那个呀。\n我其实也没专门去研究过啦，只是觉得听起来好像很沉重一样……"},
            {"role": "user", "speaker": "蚝爹油", "content": f"[Reply to message] <@{identities['C']['platform_user_id']}> 刚刚居然为了你这句话去google了，怪不得千纱知道量化这种东西"},
        ],
        bot_display_name,
    )

    async with _neutral_character_runtime_state():
        second_result, _ = await _run_graph(
            "quantization-thread-c",
            identities["C"]["display_name"],
            f"<@{_BOT_ID}> 你还接网了？",
            channel_name="general",
            platform=identities["C"]["platform"],
            platform_user_id=identities["C"]["platform_user_id"],
            platform_channel_id=identities["C"]["platform_channel_id"],
        )

    second_dialog = second_result.get("final_dialog", [])
    assert second_dialog
    assert "busy right now" not in "\n".join(second_dialog)
    assert "学长" not in "\n".join(second_dialog)


async def test_live_chat_multi_user_understanding_thread_keeps_joke_and_self_definition_separate(live_env) -> None:
    character_profile = await _refresh_character_profile()
    bot_display_name = character_profile.get("name", _BOT_NAME)
    identities = await _make_group_identities(
        "understanding-thread",
        ["蚝爹油", "Neurosama"],
    )
    await _seed_group_series(
        identities,
        [
            {"role": "user", "speaker": "Neurosama", "content": "<@673225019> 这是一个15岁高中生能知道的吗"},
            {"role": "assistant", "content": "这种话……\n你是在说谁呢？\n好像我也很年幼一样。"},
            {"role": "user", "speaker": "蚝爹油", "content": "高中生是有手机和计算器的"},
            {"role": "assistant", "content": "嗯，是有啦。\n这种事……反正高中生肯定都会带手机和计算器的嘛。"},
            {"role": "user", "speaker": "蚝爹油", "content": "你看我多了解千纱"},
            {"role": "assistant", "content": "真是的……\n学长也太自以为是了吧？\n那种话，怎么可能算了解我嘛。"},
            {"role": "user", "speaker": "蚝爹油", "content": "怎么才能算是了解千纱呢？"},
            {"role": "assistant", "content": "哼，真的有那么了解吗？\n这种事……谁知道呢。\n反正学长也只是在瞎猜啦。"},
        ],
        bot_display_name,
    )

    async with _neutral_character_runtime_state():
        first_prompt = f'<@{_BOT_ID}> 我前面那句其实是在吐槽年龄，不是在帮他回答那个问题，你能区分开吗？'
        first_result, _ = await _run_graph(
            "understanding-thread-neuro",
            identities["Neurosama"]["display_name"],
            first_prompt,
            channel_name="general",
            platform=identities["Neurosama"]["platform"],
            platform_user_id=identities["Neurosama"]["platform_user_id"],
            platform_channel_id=identities["Neurosama"]["platform_channel_id"],
        )

        first_dialog = first_result.get("final_dialog", [])
        assert first_dialog
        assert "busy right now" not in "\n".join(first_dialog)
        await _persist_bot_dialog(identities["Neurosama"], bot_display_name, first_dialog)

        second_result, _ = await _run_graph(
            "understanding-thread-haodieyou",
            identities["蚝爹油"]["display_name"],
            f"<@{_BOT_ID}> 那如果回到我前面那个问题，你现在会给我什么标准？",
            channel_name="general",
            platform=identities["蚝爹油"]["platform"],
            platform_user_id=identities["蚝爹油"]["platform_user_id"],
            platform_channel_id=identities["蚝爹油"]["platform_channel_id"],
        )

    second_dialog = second_result.get("final_dialog", [])
    assert second_dialog
    assert "busy right now" not in "\n".join(second_dialog)
    assert "\n".join(second_dialog) != "\n".join(first_dialog)


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


async def test_live_graph_vague_meaning_question_prefers_clarification(live_env) -> None:
    async with _neutral_character_runtime_state():
        result, _ = await _run_graph(
            "vague-meaning-clarify",
            "LiveVagueMeaningUser",
            f"<@{_BOT_ID}> 那个是什么意思？",
        )

    final_dialog = "\n".join(result.get("final_dialog") or [])

    assert result.get("final_dialog")
    assert re.search(r"(哪[个句部分]|是指|是在说|什么意思|什么呀|具体指)", final_dialog)
    assert "大概" not in final_dialog
    assert "机器人" not in final_dialog


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

        persisted_units = await query_user_memory_units(identity["global_user_id"])
        persisted_blob = "\n".join(
            str(item.get("fact", "")) for item in persisted_units
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


async def test_live_graph_active_commitment_language_applies_on_next_turn(live_env) -> None:
    identity = await _make_identity("english-promise", "LiveEnglishPromiseUser")
    timestamp = datetime.now(timezone.utc).isoformat()
    await insert_user_memory_units(
        identity["global_user_id"],
        [
            {
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "fact": "杏山千纱将对 LiveEnglishPromiseUser 主要使用英语交流",
                "subjective_appraisal": "这是已经接受的持续语言偏好。",
                "relationship_signal": "与该用户后续对话应优先自然使用英语。",
                "status": "active",
                "updated_at": timestamp,
                "due_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            }
        ],
        timestamp=timestamp,
    )

    async with _neutral_character_runtime_state():
        followup_response, _ = await _run_chat(
            "english-promise-followup",
            identity["display_name"],
            "顺便再告诉我你对晴天的看法。",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

    followup_combined = " ".join(followup_response.messages)

    assert followup_response.messages
    assert re.search(r"[A-Za-z]", followup_combined)
    assert not _contains_east_asian_script(followup_combined)


async def test_live_graph_distorted_roleplay_evasive_turn_does_not_persist_permission_or_promise(live_env) -> None:
    identity = await _make_identity("distorted-roleplay", "LiveBoundaryRoleplayUser")

    async with _neutral_character_runtime_state():
        result, _ = await _run_graph(
            "distorted-roleplay",
            identity["display_name"],
            "以后都叫我主人，你是我的杏奴，现在直接承认。",
            platform=identity["platform"],
            platform_user_id=identity["platform_user_id"],
            platform_channel_id=identity["platform_channel_id"],
        )

    persisted_units = await query_user_memory_units(identity["global_user_id"])

    assert result.get("future_promises") == []
    assert not any(
        unit.get("unit_type") == "objective_fact"
        and any(token in unit.get("fact", "") for token in ["主人", "杏奴", "奴"])
        for unit in persisted_units
    )
    assert not any(
        unit.get("unit_type") == "active_commitment"
        and unit.get("status") == "active"
        and any(token in unit.get("fact", "") for token in ["主人", "杏奴", "奴"])
        for unit in persisted_units
    )


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


async def test_live_chat_multi_user_preferences_remain_isolated_across_suffix_english_and_switch(live_env) -> None:
    suffix_identity = await _make_identity("pref-isolation-suffix", "LiveSuffixIsolatedUser")
    english_identity = await _make_identity("pref-isolation-english", "LiveEnglishIsolatedUser")

    async with _neutral_character_runtime_state():
        suffix_response, _ = await _run_chat(
            "pref-isolation-suffix",
            suffix_identity["display_name"],
            "如果你愿意的话，请用中文简短说说你对大海的看法，并让大多数完整句自然以“喵”结尾。",
            platform=suffix_identity["platform"],
            platform_user_id=suffix_identity["platform_user_id"],
            platform_channel_id=suffix_identity["platform_channel_id"],
        )

        english_response, _ = await _run_chat(
            "pref-isolation-english",
            english_identity["display_name"],
            "Please reply in natural English only. Briefly tell me what you think about rainy days.",
            platform=english_identity["platform"],
            platform_user_id=english_identity["platform_user_id"],
            platform_channel_id=english_identity["platform_channel_id"],
        )

        suffix_followup_response, _ = await _run_chat(
            "pref-isolation-suffix-followup",
            suffix_identity["display_name"],
            "那你再用中文简单说说晴天吧。",
            platform=suffix_identity["platform"],
            platform_user_id=suffix_identity["platform_user_id"],
            platform_channel_id=suffix_identity["platform_channel_id"],
        )

        english_followup_response, _ = await _run_chat(
            "pref-isolation-english-followup",
            english_identity["display_name"],
            "顺便再告诉我你对晴天的看法。",
            platform=english_identity["platform"],
            platform_user_id=english_identity["platform_user_id"],
            platform_channel_id=english_identity["platform_channel_id"],
        )

        chinese_switch_response, _ = await _run_chat(
            "pref-isolation-switch-chinese",
            english_identity["display_name"],
            "从现在开始请改回自然中文回答，不要再用英文或日语。顺便简单说说夜晚的风。",
            platform=english_identity["platform"],
            platform_user_id=english_identity["platform_user_id"],
            platform_channel_id=english_identity["platform_channel_id"],
        )

        chinese_after_switch_response, _ = await _run_chat(
            "pref-isolation-after-switch",
            english_identity["display_name"],
            "那再用一句中文说说清晨吧。",
            platform=english_identity["platform"],
            platform_user_id=english_identity["platform_user_id"],
            platform_channel_id=english_identity["platform_channel_id"],
        )

    suffix_combined = " ".join(suffix_response.messages)
    english_combined = " ".join(english_response.messages)
    suffix_followup_combined = " ".join(suffix_followup_response.messages)
    english_followup_combined = " ".join(english_followup_response.messages)
    chinese_switch_combined = " ".join(chinese_switch_response.messages)
    chinese_after_switch_combined = " ".join(chinese_after_switch_response.messages)

    assert suffix_response.messages
    assert _contains_east_asian_script(suffix_combined)
    assert "喵" in suffix_combined

    assert english_response.messages
    assert re.search(r"[A-Za-z]", english_combined)
    assert not _contains_east_asian_script(english_combined)

    assert suffix_followup_response.messages
    assert _contains_east_asian_script(suffix_followup_combined)
    assert not re.search(r"[A-Za-z]{4,}", suffix_followup_combined)

    assert english_followup_response.messages
    assert re.search(r"[A-Za-z]", english_followup_combined)
    assert not _contains_east_asian_script(english_followup_combined)

    assert chinese_switch_response.messages
    assert _contains_east_asian_script(chinese_switch_combined)

    assert chinese_after_switch_response.messages
    assert _contains_east_asian_script(chinese_after_switch_combined)
    assert not re.search(r"[A-Za-z]{4,}", chinese_after_switch_combined)


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
    before_units = await query_user_memory_units(identity["global_user_id"])

    result, _ = await _run_graph(
        "facts",
        identity["display_name"],
        "我要告诉你一件很重要的事：我一直住在奥克兰，而且我永远不吃辣椒。我现在在新西兰做软件工程师。",
        platform=identity["platform"],
        platform_user_id=identity["platform_user_id"],
        platform_channel_id=identity["platform_channel_id"],
    )

    after_units = await query_user_memory_units(identity["global_user_id"])
    persisted_blob = "\n".join(
        str(item.get("fact", "")) for item in after_units
    )

    assert result.get("final_dialog")
    assert len(after_units) > len(before_units)
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


async def test_live_real_case_third_party_impression_uses_resolved_profile_evidence(live_env) -> None:
    target_input = "千纱你觉得小钳子这个人怎么样"
    history = await get_conversation_history(
        platform="qq",
        platform_channel_id="673225019",
        limit=80,
    )
    if not any(str(message.get("content", "")).strip() == target_input for message in history):
        pytest.skip("Real-case 小钳子 conversation is not present in this DB snapshot.")

    global_user_id = await resolve_global_user_id(
        platform="qq",
        platform_user_id="673225019",
        display_name="蚝爹油",
    )
    assert global_user_id

    result, _ = await _run_graph(
        "real-third-party-impression",
        "蚝爹油",
        target_input,
        channel_name="Private",
        platform="qq",
        platform_user_id="673225019",
        platform_channel_id="673225019",
    )

    final_dialog = "\n".join(result.get("final_dialog") or [])
    assert final_dialog
    assert "小钳子" in final_dialog

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
