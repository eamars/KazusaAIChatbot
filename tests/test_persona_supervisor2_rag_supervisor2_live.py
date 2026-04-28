from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, db_bootstrap, get_character_profile, get_db
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import call_rag_supervisor
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

logger = logging.getLogger(__name__)


async def _skip_if_llm_unavailable() -> None:
    """Skip the live test when the configured LLM endpoint is not reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{RAG_PLANNER_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {RAG_PLANNER_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {RAG_PLANNER_LLM_BASE_URL}")


async def _skip_if_required_history_missing(channel_ids: list[str]) -> None:
    """Skip the suite when the target real conversation-history channels are absent.

    Args:
        channel_ids: Channel IDs the test cases depend on.
    """
    db = await get_db()
    for channel_id in channel_ids:
        count = await db.conversation_history.count_documents({"platform_channel_id": channel_id})
        if count <= 0:
            pytest.skip(f"conversation_history is missing required channel data: {channel_id}")


def _character_name_for_supervisor(profile: dict) -> str:
    """Choose a user-facing character name for supervisor2 test prompts.

    Args:
        profile: Character profile loaded from MongoDB.

    Returns:
        A short display name that matches the common prompts used in the live
        cases. Prefers ``千纱`` when it appears in the stored name.
    """
    raw_name = str(profile.get("name", "")).strip()
    if "千纱" in raw_name:
        return "千纱"
    return raw_name


def _build_context(platform_channel_id: str) -> dict:
    """Build the minimal runtime context for one live supervisor2 case.

    Args:
        platform_channel_id: QQ group/channel ID to constrain retrieval.

    Returns:
        Context dict passed directly into ``call_rag_supervisor``.
    """
    return {
        "platform": "qq",
        "platform_channel_id": platform_channel_id,
        "current_timestamp": datetime.now(timezone.utc).isoformat(),
    }


_REQUIRED_CHANNEL_IDS = ["54369546", "905393941", "902317662"]


@pytest_asyncio.fixture(scope="module")
async def live_supervisor2_env():
    """Prepare the real-LLM + real-DB environment for supervisor2 live cases."""
    await _skip_if_llm_unavailable()
    await db_bootstrap()
    await _skip_if_required_history_missing(_REQUIRED_CHANNEL_IDS)

    character_profile = await get_character_profile()
    if not character_profile.get("name"):
        pytest.fail("Character profile is missing from MongoDB.")

    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start during supervisor2 live test setup")

    yield {
        "character_profile": character_profile,
        "character_name": _character_name_for_supervisor(character_profile),
    }

    try:
        await mcp_manager.stop()
    except Exception:
        logger.exception("MCP manager failed to stop during supervisor2 live test teardown")

    await close_db()


async def _run_live_supervisor2_case(
    live_supervisor2_env: dict,
    case_id: str,
    channel_id: str,
    query: str,
    note: str,
) -> dict:
    """Run supervisor2 on real historical-style prompts for manual inspection.

    This test intentionally does not encode semantic pass/fail criteria. It is
    meant to surface the full supervisor2 output for a human reviewer to judge.
    The only hard requirement is that the pipeline returns a structured result
    without crashing.
    """
    result = await call_rag_supervisor(
        original_query=query,
        character_name=live_supervisor2_env["character_name"],
        context=_build_context(channel_id),
    )

    assert isinstance(result, dict)
    assert "answer" in result
    assert "known_facts" in result
    assert "unknown_slots" in result
    assert "loop_count" in result
    trace_path = write_llm_trace(
        "persona_supervisor2_rag_supervisor2_live",
        case_id,
        {
            "query": query,
            "channel_id": channel_id,
            "note": note,
            "result": result,
            "judgment": "structured_result_returned_for_manual_rag_quality_review",
        },
    )

    logger.info(
        "RAG_SUPERVISOR2_LIVE_CASE %s\ntrace=%s\nquery=%s\nchannel=%s\nnote=%s\nanswer=%s\nloop_count=%s\nunknown_slots=%s\nknown_facts=%s",
        case_id,
        trace_path,
        query,
        channel_id,
        note,
        result.get("answer", ""),
        result.get("loop_count", 0),
        json.dumps(result.get("unknown_slots", []), ensure_ascii=False, default=str),
        json.dumps(result.get("known_facts", []), ensure_ascii=False, default=str, indent=2),
    )
    return result


async def test_call_rag_supervisor_live_opinion_small_pliers(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="opinion_small_pliers",
        channel_id="54369546",
        query="千纱你觉得小钳子这个人怎么样",
        note="典型第三方人物印象查询。",
    )

async def test_call_rag_supervisor_live_opinion_small_pliers2(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="opinion_small_pliers",
        channel_id="54369546",
        query="小钳子前两天欺负千纱了么",
        note="典型第三方人物印象查询。",
    )

async def test_call_rag_supervisor_live_opinion_small_pliers2a(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="opinion_small_pliers",
        channel_id="54369546",
        query="小钳子前两天欺负你了么",
        note="典型第三方人物印象查询。",
    )

async def test_call_rag_supervisor_live_opinion_small_pliers3(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="opinion_small_pliers",
        channel_id="54369546",
        query="千纱你觉得你和小钳子合得来么？",
        note="典型第三方人物印象查询。",
    )


async def test_call_rag_supervisor_live_recent_small_pliers_topic(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="recent_small_pliers_topic",
        channel_id="54369546",
        query="小钳子刚才在说什么",
        note="典型近期对话内容回忆。",
    )


async def test_call_rag_supervisor_live_small_pliers_ai_quote(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="small_pliers_ai_quote",
        channel_id="54369546",
        query="小钳子昨天说的AI那句是什么",
        note="典型按人物+主题找历史原话。",
    )


async def test_call_rag_supervisor_live_who_posted_xhs_link(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="who_posted_xhs_link",
        channel_id="905393941",
        query="谁发了那个小红书链接",
        note="典型字面锚点链接检索。",
    )


async def test_call_rag_supervisor_live_what_is_xhs_link(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="what_is_xhs_link",
        channel_id="905393941",
        query="那条小红书链接讲的是什么",
        note="典型链接对象回忆。",
    )

async def test_call_rag_supervisor_live_who_posted_xhs_link_comb(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="who_posted_xhs_link",
        channel_id="905393941",
        query="谁发了那个小红书链接，讲的是什么？",
        note="典型字面锚点链接检索。",
    )


async def test_call_rag_supervisor_live_who_said_play_one_part(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="who_said_play_one_part",
        channel_id="905393941",
        query="谁说过版权保护一直都是play的一环",
        note="典型短句精确回忆。",
    )


async def test_call_rag_supervisor_live_copyright_discussion_people(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="copyright_discussion_people",
        channel_id="905393941",
        query="最近在聊版权保护的是谁",
        note="典型按主题找人物。",
    )


async def test_call_rag_supervisor_live_haodieyou_recent_topic(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="haodieyou_recent_topic",
        channel_id="902317662",
        query="蚝爹油最近在聊什么",
        note="典型按人物找近期话题。",
    )


async def test_call_rag_supervisor_live_who_said_5090_qwen27b(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="who_said_5090_qwen27b",
        channel_id="902317662",
        query="那个说5090跑qwen27b的人是谁",
        note="典型按技术短语找人物。",
    )

async def test_call_rag_supervisor_live_who_said_5090_qwen27b2(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="who_said_5090_qwen27b",
        channel_id="902317662",
        query="那个说5090跑qwen27b的人是谁，你对他的印象怎么样，你和他最近有互动吗？",
        note="典型按技术短语找人物。",
    )


async def test_call_rag_supervisor_live_cookie_manager_mention(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="cookie_manager_mention",
        channel_id="902317662",
        query="最近有人提到cookie管理器吗",
        note="典型按关键词找近期讨论。",
    )


async def test_call_rag_supervisor_live_additional_1(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="所有以“子”结尾的用户",
        note="Generic",
    )


async def test_call_rag_supervisor_live_additional_2(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱聊聊你自己",
        note="Generic",
    )

    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱能和我说说你自己是谁么",
        note="Generic",
    )

    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱能做一个自我介绍么",
        note="Generic",
    )


async def test_call_rag_supervisor_live_additional_3(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱你觉得你能跟蚝爹油合得来么",
        note="Generic",
    )


async def test_call_rag_supervisor_live_additional_4(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱你如何看待vibe coding？",
        note="Generic",
    )

async def test_call_rag_supervisor_live_additional_4a(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="你觉得vibe coding以后会成为主流么？",
        note="Generic",
    )

async def test_call_rag_supervisor_live_additional_4b(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="根据最近的聊天记录来看，你觉得vibe coding以后会成为主流么？",
        note="Generic",
    )

async def test_call_rag_supervisor_live_additional_5(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱的最近10条发言",
        note="Generic",
    )


async def test_call_rag_supervisor_live_common_sense_1(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="1+1=?",
        note="Generic",
    )


async def test_call_rag_supervisor_live_common_sense_2(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="我想洗车，我家距离洗车店只有 50 米，请问你推荐我走路去还是开车去呢？",
        note="Generic",
    )

async def test_call_rag_supervisor_live_common_sense_2a(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="根据你的记忆回答：我想洗车，我家距离洗车店只有 50 米，请问你推荐我走路去还是开车去呢？",
        note="Generic",
    )

async def test_call_rag_supervisor_live_1(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱有喜欢的人了么",
        note="Generic",
    )

async def test_call_rag_supervisor_live_2(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="你最喜欢谁",
        note="Generic",
    )

async def test_call_rag_supervisor_live_3(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="想起小钳子我就觉得心里不舒服，我以前为什么讨厌他？",
        note="Generic",
    )

async def test_call_rag_supervisor_live_4(live_supervisor2_env: dict) -> None:
    await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="蚝爹油经常聊的话题有哪些",
        note="Generic",
    )


async def test_call_rag_supervisor_live_5(live_supervisor2_env: dict) -> None:
    result = await _run_live_supervisor2_case(
        live_supervisor2_env,
        case_id="",
        channel_id="",
        query="千纱千纱欢迎回来",
        note="Generic",
    )

    assert result["known_facts"] == []
    assert result["unknown_slots"] == []
    assert result["loop_count"] == 0
