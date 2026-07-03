"""Live LLM routing checks for the web_agent3 Bilibili source."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    WEB_SEARCH_LLM_API_KEY,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.web_agent3 import agent as agent_module
from kazusa_ai_chatbot.rag.web_agent3.subagent import _SUBAGENT_NAMES
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


class _CapturingLiveLLM:
    """Capture live router prompts and outputs while preserving the real call."""

    def __init__(self, wrapped_llm: Any) -> None:
        self._wrapped_llm = wrapped_llm
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: list[Any], *, config: Any) -> Any:
        """Delegate to the live model and record the prompt/output boundary."""
        response = await self._wrapped_llm.ainvoke(messages, config=config)
        self.calls.append({
            "messages": [
                {
                    "type": message.__class__.__name__,
                    "content": str(message.content),
                }
                for message in messages
            ],
            "route_name": getattr(config, "route_name", None),
            "model": getattr(config, "model", None),
            "base_url": getattr(config, "base_url", None),
            "raw_output": str(response.content),
        })
        return response


async def _run_bilibili_router_case(
    *,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    task: str,
    expected_action: str,
    expected_status: str,
    expected_content_type: str | None = None,
    expected_content_scope: str | None = None,
    expected_popularity_basis: str | None = None,
) -> dict[str, Any]:
    """Run one live router case and execute the selected source."""
    await _skip_if_live_llm_unavailable()
    assert "bilibili" in _SUBAGENT_NAMES

    live_llm = _CapturingLiveLLM(agent_module._generator_llm)
    monkeypatch.setattr(agent_module, "_generator_llm", live_llm)
    state = {
        "task": task,
        "context": {
            "original_query": task,
            "current_slot": "Bilibili public evidence request",
            "channel_topic": "web_agent3 live LLM routing verification",
        },
        "expected_response": "Return source-grounded Bilibili evidence.",
        "messages": [],
        "router_decision": {
            "action": "stop",
            "source": "web_read",
            "query": "",
        },
        "observations": [],
        "evaluator_feedback": "",
        "should_stop": False,
        "retry": 0,
        "prompt_timestamp": "2026-07-02 21:30 (Thursday)",
        "knowledge_metadata": {},
        "final_response": "",
        "final_status": "",
        "final_reason": "",
        "final_is_empty_result": False,
    }

    generator_update = await agent_module._tool_call_generator(state)
    decision = generator_update["router_decision"]
    executor_state = {
        **state,
        "router_decision": decision,
        "messages": generator_update["messages"],
    }
    executor_update = await agent_module._tool_call_executor(executor_state)
    observation_record = executor_update["observations"][-1]
    source_result = observation_record["result"]
    forbidden_fragments = _forbidden_observation_fragments(source_result)
    trace_path = write_llm_trace(
        "web_agent3_bilibili_live_llm",
        case_id,
        {
            "route": {
                "base_url": WEB_SEARCH_LLM_BASE_URL,
                "model": WEB_SEARCH_LLM_MODEL,
            },
            "input_state": state,
            "router_prompt_call": live_llm.calls[-1],
            "router_decision": decision,
            "source_observation": observation_record,
            "forbidden_observation_fragments": forbidden_fragments,
            "judgment": "manual_review_required_for_bilibili_route_quality",
        },
    )
    logger.info(
        "WEB_AGENT3_BILIBILI_LIVE case=%s trace=%s decision=%s result_status=%s",
        case_id,
        trace_path,
        json.dumps(decision, ensure_ascii=True),
        source_result.get("status"),
    )

    assert decision["source"] == "bilibili"
    assert decision["action"] == expected_action
    assert source_result["source"] == "bilibili"
    assert source_result["status"] == expected_status
    if expected_content_type is not None:
        assert source_result["content_type"] == expected_content_type
    if expected_content_scope is not None:
        assert source_result["content_scope"] == expected_content_scope
    if expected_popularity_basis is not None:
        assert source_result["popularity_basis"] == expected_popularity_basis
    assert forbidden_fragments == []
    return {
        "trace_path": str(trace_path),
        "decision": decision,
        "source_result": source_result,
    }


async def _skip_if_live_llm_unavailable() -> None:
    """Skip when the configured web-search LLM endpoint is unavailable."""
    headers = {}
    if WEB_SEARCH_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {WEB_SEARCH_LLM_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{WEB_SEARCH_LLM_BASE_URL.rstrip('/')}/models",
                headers=headers,
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            "WEB_SEARCH_LLM endpoint is unavailable: "
            f"{WEB_SEARCH_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            "WEB_SEARCH_LLM endpoint returned server error "
            f"{response.status_code}: {WEB_SEARCH_LLM_BASE_URL}"
        )


def _forbidden_observation_fragments(source_result: object) -> list[str]:
    """Return source material that must stay out of prompt-facing evidence."""
    serialized = json.dumps(source_result, ensure_ascii=False, default=str).lower()
    errors: list[str] = []
    for fragment in (
        "credential",
        "authorization",
        "cookie",
        "download",
        "subtitle_url",
        "comments",
        "headers",
    ):
        if fragment in serialized:
            errors.append(fragment)

    return errors


async def test_web_agent3_bilibili_live_routes_video_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live router should send a Bilibili video URL to Bilibili read."""
    result = await _run_bilibili_router_case(
        monkeypatch=monkeypatch,
        case_id="video_read_bv_url",
        task=(
            '请读取这个 Bilibili 视频并告诉我它讲了什么：'
            'https://www.bilibili.com/video/BV1CqV266EJY/'
        ),
        expected_action="read",
        expected_status="success",
        expected_content_type="video",
        expected_content_scope="video",
    )

    source_result = result["source_result"]
    assert source_result["public_id"] == "BV1CqV266EJY"
    assert source_result["title"]
    assert "metadata" in source_result["content_basis"]
    assert "pages" in source_result["content_basis"]


async def test_web_agent3_bilibili_live_routes_article_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live router should send a Bilibili article URL to Bilibili read."""
    result = await _run_bilibili_router_case(
        monkeypatch=monkeypatch,
        case_id="article_read_cv_url",
        task=(
            '请读取这个 Bilibili 专栏文章并说明它是什么内容：'
            'https://www.bilibili.com/read/cv44952180'
        ),
        expected_action="read",
        expected_status="success",
        expected_content_type="article",
        expected_content_scope="article",
    )

    source_result = result["source_result"]
    assert source_result["public_id"] == "cv44952180"
    assert source_result["title"]
    assert source_result["stats_summary"]


async def test_web_agent3_bilibili_live_routes_general_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live router should send untyped Bilibili search to Bilibili search."""
    result = await _run_bilibili_router_case(
        monkeypatch=monkeypatch,
        case_id="general_search_python_tutorial",
        task='帮我在 Bilibili 上搜索 Python 教程，找几个可参考的结果。',
        expected_action="search",
        expected_status="success",
        expected_content_scope="general",
        expected_popularity_basis="provider_default",
    )

    source_result = result["source_result"]
    assert source_result["results"]
    assert source_result["results"][0]["title"]


async def test_web_agent3_bilibili_live_routes_popular_video_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live router should preserve popular-video Bilibili search intent."""
    result = await _run_bilibili_router_case(
        monkeypatch=monkeypatch,
        case_id="popular_video_search_vibe_coding",
        task='帮我在bilibili上搜索关于vibe coding相关视频并且推荐给我一个最热门的视频',
        expected_action="search",
        expected_status="success",
        expected_content_scope="video",
        expected_popularity_basis="most_clicked",
    )

    source_result = result["source_result"]
    assert source_result["results"]
    assert source_result["results"][0]["content_type"] == "video"


async def test_web_agent3_bilibili_live_routes_unknown_bilibili_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live router should keep unsupported Bilibili URLs inside Bilibili."""
    result = await _run_bilibili_router_case(
        monkeypatch=monkeypatch,
        case_id="unsupported_bilibili_url",
        task='请读取这个 Bilibili 链接：https://www.bilibili.com/unknown/123',
        expected_action="read",
        expected_status="unsupported",
        expected_content_type="unknown",
        expected_content_scope="unknown",
    )

    source_result = result["source_result"]
    assert source_result["message"] == (
        "Unsupported or unrecognized Bilibili public target."
    )
