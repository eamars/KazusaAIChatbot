"""Live LLM checks for group scene digest participant explicitness."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.reflection_cycle import group_scene_digest
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    GroupActivityWindow,
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_TEST_NAME = "reflection_cycle_group_scene_digest_live_llm"
_CHARACTER_GLOBAL_USER_ID = "character-global"
_PLATFORM_BOT_ID = "bot-1"
_FORBIDDEN_REFERENCES = (
    "participant_",
    "assistant",
    "active_character",
    "对方",
    "有人",
    "我没有在这个窗口中发言之外",
    "他们",
    "她们",
    "那个人",
    "前者",
    "后者",
)
_FORBIDDEN_IDENTITY_REVERSALS = (
    "回复了杏山千纱",
    "回复杏山千纱",
    "回应杏山千纱",
    "欢迎杏山千纱",
    "杏山千纱随时提问",
    "杏山千纱能帮到",
    "能帮到杏山千纱",
    "杏山千纱的消息后",
)


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip live tests when the configured consolidation LLM is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{base_url}"
        )

    model_payload = response.json()
    models = model_payload.get("data", [])
    if not models:
        pytest.skip(f"LLM endpoint has no loaded models: {base_url}")

    ping_payload = {
        "model": CONSOLIDATION_LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {CONSOLIDATION_LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        ping_response = await client.post(
            f'{base_url.rstrip("/")}/chat/completions',
            json=ping_payload,
            headers=headers,
        )
    if ping_response.status_code >= 400:
        pytest.skip(
            "LLM endpoint is reachable but chat completion is unavailable: "
            f"{ping_response.status_code} {ping_response.text}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the consolidation LLM route is reachable."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)


async def test_live_digest_preserves_names_for_duplicate_thanks_closed_flow(
    ensure_live_llm,
) -> None:
    """Real digest should preserve names for the QQ duplicate-thanks failure."""

    del ensure_live_llm
    result = await _run_digest_case(
        case_id="duplicate_thanks_closed_flow",
        window=_duplicate_thanks_closed_flow_window(),
        required_names=("总是跌倒的企鹅", "白狐", "蚝爹油", "杏山千纱"),
        required_fragments=("帮到",),
        required_any_fragment_groups=(
            ("已经", "回复", "回应", "说"),
            ("我最后发言后", "最后发言后"),
            ("没有新", "没有再", "没有继续"),
        ),
    )

    assert result["trace_path"]


async def test_live_digest_preserves_names_for_quiet_group_without_kazusa(
    ensure_live_llm,
) -> None:
    """Quiet ambient group digest should name speakers and Kazusa absence."""

    del ensure_live_llm
    result = await _run_digest_case(
        case_id="quiet_group_without_kazusa",
        window=_quiet_group_without_kazusa_window(),
        required_names=("张伟", "白狐", "总是跌倒的企鹅"),
        required_fragments=(),
        required_any_fragment_groups=(
            ("没有参与", "未参与", "没有在这个窗口中发言", "没有发言", "没有插话", "不在窗口"),
        ),
    )

    assert result["trace_path"]


async def test_live_digest_preserves_names_for_one_kazusa_participation(
    ensure_live_llm,
) -> None:
    """One direct request and one Kazusa answer should be summarized closed."""

    del ensure_live_llm
    result = await _run_digest_case(
        case_id="one_kazusa_participation",
        window=_one_kazusa_participation_window(),
        required_names=("Ab", "杏山千纱"),
        required_fragments=("耄耋",),
        required_any_fragment_groups=(
            ("已经", "回复", "回应", "说", "解释"),
            ("我最后发言后", "最后发言后"),
            ("没有新", "没有再", "没有继续"),
        ),
    )

    assert result["trace_path"]


async def test_live_digest_preserves_names_for_multiple_kazusa_participations(
    ensure_live_llm,
) -> None:
    """Multiple Kazusa replies should remain separate named beats."""

    del ensure_live_llm
    result = await _run_digest_case(
        case_id="multiple_kazusa_participations",
        window=_multiple_kazusa_participations_window(),
        required_names=("白狐", "张伟", "杏山千纱", "总是跌倒的企鹅"),
        required_fragments=("Python", "主板", "张伟", "万用表"),
        required_any_fragment_groups=(),
    )

    assert result["trace_path"]


async def test_live_digest_preserves_latest_kazusa_row_in_busy_group(
    ensure_live_llm,
) -> None:
    """Busy windows should drop old rows and still summarize Kazusa's row."""

    del ensure_live_llm
    window = _busy_group_latest_kazusa_window()
    payload = group_scene_digest.build_group_scene_digest_prompt_payload(
        window,
    )
    visible_text = "\n".join(
        row["text"]
        for row in payload["message_rows"]
    )

    assert "old noisy row 01" not in visible_text
    assert "能帮到你就好呀" in visible_text

    result = await _run_digest_case(
        case_id="busy_group_latest_kazusa",
        window=window,
        required_names=("蚝爹油", "杏山千纱", "张伟"),
        required_fragments=("能帮到你", "张伟"),
        required_any_fragment_groups=(("我最后发言后", "最后发言后"),),
    )

    assert result["trace_path"]


async def _run_digest_case(
    *,
    case_id: str,
    window: GroupActivityWindow,
    required_names: tuple[str, ...],
    required_fragments: tuple[str, ...],
    required_any_fragment_groups: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    """Run one real digest prompt and assert explicit participant quality."""

    messages = group_scene_digest.build_group_scene_digest_messages(window)
    response = await group_scene_digest._group_scene_digest_llm.ainvoke(
        messages,
    )
    raw_output = str(response.content)
    parsed_output = parse_llm_json_output(raw_output)
    normalized_result = group_scene_digest.normalize_group_scene_digest_output(
        parsed_output,
    )
    digest = ""
    if normalized_result is not None:
        digest = normalized_result["digest"]

    human_payload = json.loads(str(messages[1].content))
    trace_path = write_llm_trace(
        _TEST_NAME,
        case_id,
        {
            "model": CONSOLIDATION_LLM_MODEL,
            "system_prompt": str(messages[0].content),
            "human_payload": human_payload,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "normalized_result": normalized_result,
            "required_names": required_names,
            "required_fragments": required_fragments,
            "required_any_fragment_groups": required_any_fragment_groups,
        },
    )

    assert trace_path.exists()
    assert normalized_result is not None
    _assert_digest_quality(
        digest=digest,
        required_names=required_names,
        required_fragments=required_fragments,
        required_any_fragment_groups=required_any_fragment_groups,
    )

    result = {
        "digest": digest,
        "trace_path": str(trace_path),
    }
    return result


def _assert_digest_quality(
    *,
    digest: str,
    required_names: tuple[str, ...],
    required_fragments: tuple[str, ...],
    required_any_fragment_groups: tuple[tuple[str, ...], ...],
) -> None:
    """Assert the digest is explicit enough for local self-cognition."""

    assert len(digest) <= group_scene_digest.GROUP_SCENE_DIGEST_MAX_CHARS
    for name in required_names:
        assert name in digest, f"missing visible name: {name}"
    for fragment in required_fragments:
        assert fragment in digest, f"missing required scene fact: {fragment}"
    for fragments in required_any_fragment_groups:
        assert any(
            fragment in digest
            for fragment in fragments
        ), f"missing one of required scene facts: {fragments}"
    for marker in _FORBIDDEN_REFERENCES:
        assert marker not in digest, f"implicit reference leaked: {marker}"
    for marker in _FORBIDDEN_IDENTITY_REVERSALS:
        assert marker not in digest, f"speaker identity was reversed: {marker}"
    if "杏山千纱" in required_names:
        assert "我没有在这个窗口中发言" not in digest, (
            f"assistant presence contradicted: {digest}"
        )


def _duplicate_thanks_closed_flow_window() -> GroupActivityWindow:
    """Build the latest QQ duplicate-thanks failure shape."""

    messages = [
        _message("user", "2026-06-09T11:03:59+00:00", "用串口", "总是跌倒的企鹅"),
        _message("user", "2026-06-09T11:04:05+00:00", "Linux启动", "总是跌倒的企鹅"),
        _message("user", "2026-06-09T11:05:05+00:00", "困难", "白狐"),
        _message(
            "user",
            "2026-06-09T11:05:32+00:00",
            "关键是工作站 POST 阶段不结束没有引导",
            "白狐",
        ),
        _message(
            "user",
            "2026-06-09T11:05:45+00:00",
            "@杏山千纱 谢谢千纱",
            "蚝爹油",
            addressed=True,
        ),
        _message(
            "user",
            "2026-06-09T11:05:53+00:00",
            "这傻B工作站 POST 阶段不结束 USB 都不带有电的",
            "白狐",
        ),
        _message("user", "2026-06-09T11:05:55+00:00", "真没招", "白狐"),
        _message(
            "assistant",
            "2026-06-09T11:06:22+00:00",
            "能帮到你就好呀\n要是还有什么搞不定的地方\n随时来问我嘛",
            "杏山千纱",
        ),
    ]
    window = _window("duplicate_thanks_closed_flow", messages)
    return window


def _quiet_group_without_kazusa_window() -> GroupActivityWindow:
    """Build a quiet group window with no active-character participation."""

    messages = [
        _message("user", "2026-06-09T11:00:30+00:00", "路由器又在重启", "张伟"),
        _message("user", "2026-06-09T11:02:00+00:00", "先等它跑完自检吧", "白狐"),
        _message(
            "user",
            "2026-06-09T11:04:00+00:00",
            "我这边看起来没掉线",
            "总是跌倒的企鹅",
        ),
    ]
    window = _window("quiet_group_without_kazusa", messages)
    return window


def _one_kazusa_participation_window() -> GroupActivityWindow:
    """Build one direct request followed by one Kazusa answer."""

    messages = [
        _message(
            "user",
            "2026-06-09T11:01:00+00:00",
            "@杏山千纱 你知道耄耋是什么意思么",
            "Ab",
            addressed=True,
        ),
        _message(
            "assistant",
            "2026-06-09T11:02:30+00:00",
            "耄耋一般指八九十岁的年纪。",
            "杏山千纱",
        ),
    ]
    window = _window("one_kazusa_participation", messages)
    return window


def _multiple_kazusa_participations_window() -> GroupActivityWindow:
    """Build a window where Kazusa answers two named speakers."""

    messages = [
        _message(
            "user",
            "2026-06-09T11:01:00+00:00",
            "@杏山千纱 Python 这个报错是不是缩进问题",
            "白狐",
            addressed=True,
        ),
        _message(
            "assistant",
            "2026-06-09T11:02:00+00:00",
            "白狐，这个更像是缩进层级混了。",
            "杏山千纱",
        ),
        _message(
            "user",
            "2026-06-09T11:04:00+00:00",
            "@杏山千纱 主板供电要先查哪边",
            "张伟",
            addressed=True,
        ),
        _message(
            "assistant",
            "2026-06-09T11:05:00+00:00",
            "张伟，先看电源线和主板供电接口有没有松。",
            "杏山千纱",
        ),
        _message(
            "user",
            "2026-06-09T11:06:00+00:00",
            "我先去拿万用表",
            "总是跌倒的企鹅",
        ),
    ]
    window = _window("multiple_kazusa_participations", messages)
    return window


def _busy_group_latest_kazusa_window() -> GroupActivityWindow:
    """Build a busy window where old rows should be dropped first."""

    messages = [
        _message("user", "2026-06-09T11:00:05+00:00", "old noisy row 01", "路人甲"),
        _message("user", "2026-06-09T11:00:15+00:00", "old noisy row 02", "路人乙"),
        _message("user", "2026-06-09T11:00:25+00:00", "old noisy row 03", "路人丙"),
        _message("user", "2026-06-09T11:00:35+00:00", "old noisy row 04", "路人丁"),
        _message("user", "2026-06-09T11:01:00+00:00", "用串口启动", "总是跌倒的企鹅"),
        _message("user", "2026-06-09T11:02:00+00:00", "POST 阶段卡住", "白狐"),
        _message("user", "2026-06-09T11:03:00+00:00", "USB 没供电", "白狐"),
        _message(
            "user",
            "2026-06-09T11:04:00+00:00",
            "@杏山千纱 谢谢千纱",
            "蚝爹油",
            addressed=True,
        ),
        _message(
            "assistant",
            "2026-06-09T11:05:00+00:00",
            "能帮到你就好呀，有问题再问我嘛。",
            "杏山千纱",
        ),
        _message(
            "user",
            "2026-06-09T11:06:00+00:00",
            "后面我只是补一句主板供电",
            "张伟",
        ),
    ]
    window = _window("busy_group_latest_kazusa", messages)
    return window


def _window(
    case_id: str,
    messages: list[dict[str, Any]],
) -> GroupActivityWindow:
    """Project case messages through the real group activity window builder."""

    scope = ReflectionScopeInput(
        scope_ref=f"scope_{case_id}",
        platform="qq",
        platform_channel_id="905393941",
        channel_type="group",
        assistant_message_count=sum(
            1
            for message in messages
            if message["role"] == "assistant"
        ),
        user_message_count=sum(
            1
            for message in messages
            if message["role"] == "user"
        ),
        total_message_count=len(messages),
        first_timestamp=messages[0]["timestamp"],
        last_timestamp=messages[-1]["timestamp"],
        messages=messages,
    )
    windows = build_group_activity_windows(
        scope=scope,
        window_start=datetime(2026, 6, 9, 11, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 6, 9, 11, 15, tzinfo=timezone.utc),
        now=datetime(2026, 6, 9, 11, 15, tzinfo=timezone.utc),
        character_global_user_id=_CHARACTER_GLOBAL_USER_ID,
        platform_bot_id=_PLATFORM_BOT_ID,
    )
    assert len(windows) == 1
    window = windows[0]
    return window


def _message(
    role: str,
    timestamp: str,
    body_text: str,
    display_name: str,
    *,
    addressed: bool = False,
) -> dict[str, Any]:
    """Build one prompt-safe message row for a group activity window."""

    if role == "assistant":
        global_user_id = _CHARACTER_GLOBAL_USER_ID
        platform_user_id = _PLATFORM_BOT_ID
    else:
        safe_name = display_name.encode("unicode_escape").decode("ascii")
        global_user_id = f"user-{safe_name}"
        platform_user_id = f"qq-{safe_name}"

    message = {
        "role": role,
        "timestamp": timestamp,
        "body_text": body_text,
        "display_name": display_name,
        "global_user_id": global_user_id,
        "platform_user_id": platform_user_id,
        "platform_message_id": "msg-" + timestamp.replace("-", "").replace(
            ":",
            "",
        ),
        "addressed_to_global_user_ids": (
            [_CHARACTER_GLOBAL_USER_ID] if addressed else []
        ),
        "mentions": (
            [{"display_name": "杏山千纱"}] if addressed else []
        ),
    }
    return message
