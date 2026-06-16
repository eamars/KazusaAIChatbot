import json

import httpx
import pytest

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import AFFINITY_DEFAULT, JSON_REPAIR_LLM_BASE_URL
from kazusa_ai_chatbot import utils as utils_module
from kazusa_ai_chatbot.utils import (
    build_affinity_block,
    parse_llm_json_output,
)
from tests.llm_trace import write_llm_trace


_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT = """{
  "tool_calls": [
    {
      "tool": "tool name",
      "args": {
        "parameter": "value",
        "target_channel_type": "group | private when target_channel is not same"
      }
    }
  ]
}"""


class _CapturingRepairLLM:
    """Wrap the live JSON-repair LLM and retain the most recent exchange."""

    def __init__(self, inner_llm):
        self.inner_llm = inner_llm
        self.messages = None
        self.raw_output = None

    def invoke(self, messages, *, config):
        self.messages = messages
        response = self.inner_llm.invoke(messages, config=config)
        self.raw_output = str(response.content)
        return_value = response
        return return_value


def _skip_if_json_repair_llm_unavailable() -> None:
    """Skip live JSON repair tests when the configured endpoint is unavailable."""

    try:
        response = httpx.get(
            f"{JSON_REPAIR_LLM_BASE_URL.rstrip('/')}/models",
            timeout=3.0,
        )
    except httpx.HTTPError as exc:
        pytest.skip(f"JSON repair LLM endpoint is unavailable: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"JSON repair LLM endpoint returned server error "
            f"{response.status_code}: {JSON_REPAIR_LLM_BASE_URL}"
        )


@pytest.fixture()
def ensure_json_repair_live_llm() -> None:
    """Ensure the configured JSON repair LLM endpoint is reachable."""

    _skip_if_json_repair_llm_unavailable()


def _write_json_repair_trace(
    case_id: str,
    *,
    broken_json: str,
    expected_output_format: str,
    raw_repair_output: str | None,
    parsed_output: dict,
) -> None:
    """Persist one live JSON repair trace for manual inspection."""

    trace_path = write_llm_trace(
        "json_repair_live_llm",
        case_id,
        {
            "broken_json": broken_json,
            "expected_output_format": expected_output_format,
            "raw_repair_output": raw_repair_output,
            "parsed_output": parsed_output,
            "judgment": "parsed_output_matches_expected_contract_when_test_passes",
        },
    )
    print(f"json_repair_live_trace={trace_path}")


def test_trim_history_dict():
    """Test the trim_history_dict function."""
    from kazusa_ai_chatbot.utils import trim_history_dict
    
    history = [
        {
            "display_name": "<user A>",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "Hello",
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t1",
        },
        {
            "display_name": "<user B>",
            "platform_user_id": "user_456",
            "global_user_id": "uuid-2",
            "body_text": "Hi",
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t2",
        },
    ]
    
    trimmed = trim_history_dict(history)
    assert len(trimmed) == 2
    assert trimmed[0]["name"] == "<user A>"
    assert trimmed[0]["platform_user_id"] == "user_123"
    assert trimmed[0]["body_text"] == "Hello"
    assert "content" not in trimmed[0]
    assert trimmed[0]["role"] == "user"
    assert trimmed[1]["name"] == "<user B>"
    assert trimmed[1]["platform_user_id"] == "user_456"
    assert trimmed[1]["body_text"] == "Hi"
    assert "content" not in trimmed[1]
    assert trimmed[1]["role"] == "user"


def test_trim_history_dict_projects_image_only_attachment_text():
    """Image-only history rows should expose prompt-safe image meaning."""
    from kazusa_ai_chatbot.utils import trim_history_dict

    history = [
        {
            "display_name": "总是跌倒的企鹅",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "",
            "raw_wire_text": "[CQ:image,file=sam.png]",
            "attachments": [
                {
                    "media_type": "image/png",
                    "description": "拓竹入驻山姆，不只是上架 3D 打印机",
                },
            ],
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t1",
        },
    ]

    trimmed = trim_history_dict(history)

    assert trimmed[0]["body_text"] == (
        "<image>拓竹入驻山姆，不只是上架 3D 打印机</image>"
    )


def test_trim_history_dict_appends_image_block_after_body_text():
    """Authored text should remain before appended image blocks."""
    from kazusa_ai_chatbot.utils import trim_history_dict

    history = [
        {
            "display_name": "总是跌倒的企鹅",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "这个也进山姆了",
            "attachments": [
                {
                    "media_kind": "image",
                    "description": "拓竹入驻山姆，不只是上架 3D 打印机",
                },
            ],
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t1",
        },
    ]

    trimmed = trim_history_dict(history)

    assert trimmed[0]["body_text"] == (
        "这个也进山姆了\n"
        "<image>拓竹入驻山姆，不只是上架 3D 打印机</image>"
    )


def test_trim_history_dict_projects_reply_image_description():
    """Reply image descriptions should survive in reply excerpt text."""
    from kazusa_ai_chatbot.utils import trim_history_dict

    history = [
        {
            "display_name": "总是跌倒的企鹅",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "小孩的消费力还是强",
            "attachments": [],
            "reply_context": {
                "reply_to_message_id": "1581464756",
                "reply_to_platform_user_id": "user_123",
                "reply_to_display_name": "总是跌倒的企鹅",
                "reply_attachments": [
                    {
                        "media_kind": "image",
                        "description": "拓竹入驻山姆，不只是上架 3D 打印机",
                    },
                ],
            },
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t2",
        },
    ]

    trimmed = trim_history_dict(history)

    assert trimmed[0]["reply_context"]["reply_excerpt"] == (
        "<image>拓竹入驻山姆，不只是上架 3D 打印机</image>"
    )


def test_trim_history_dict_escapes_image_description_boundaries():
    """Image descriptions should not create or close image tags."""
    from kazusa_ai_chatbot.utils import trim_history_dict

    history = [
        {
            "display_name": "User",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "",
            "attachments": [
                {
                    "media_kind": "image",
                    "description": "A < B & already </image> closed",
                },
            ],
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t1",
        },
    ]

    trimmed = trim_history_dict(history)

    assert trimmed[0]["body_text"] == (
        "<image>A &lt; B &amp; already &lt;/image&gt; closed</image>"
    )


def test_trim_history_dict_truncates_long_image_description():
    """Image descriptions should reuse the current prompt attachment cap."""
    from kazusa_ai_chatbot.message_envelope import (
        MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS,
    )
    from kazusa_ai_chatbot.utils import trim_history_dict

    history = [
        {
            "display_name": "User",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "",
            "attachments": [
                {
                    "media_kind": "image",
                    "description": "x" * (
                        MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS + 10
                    ),
                },
            ],
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t1",
        },
    ]

    trimmed = trim_history_dict(history)
    projected_description = trimmed[0]["body_text"].removeprefix(
        "<image>",
    ).removesuffix("</image>")

    assert len(projected_description) == MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS
    assert projected_description.endswith("...")


def test_parse_llm_json_output_accepts_markdown_fenced_raw_output():
    """Markdown fences are repaired from raw LLM text without escaped wrapping."""
    raw_output = """```json
{
  "continuity": "related_shift",
  "open_loops": ["follow up"]
}
```"""

    result = parse_llm_json_output(raw_output)

    assert result == {
        "continuity": "related_shift",
        "open_loops": ["follow up"],
    }


def test_parse_llm_json_output_returns_empty_dict_for_repaired_list(monkeypatch):
    """Non-object repair results should fail closed to an empty object."""

    def _list_repair(_broken_string: str, *, expected_output_format=None):
        del expected_output_format
        return []

    monkeypatch.setattr(utils_module, "parse_json_with_llm", _list_repair)

    result = parse_llm_json_output("[]")

    assert result == {}


def test_parse_llm_json_output_does_not_expose_global_trace_state():
    """The shared parser must not expose mutable last-call trace state."""

    assert not hasattr(utils_module, "_LAST_PARSE_LLM_JSON_OUTPUT_TRACE")
    assert not hasattr(utils_module, "get_last_parse_llm_json_output_trace")


def test_parse_llm_json_output_rejects_non_string_expected_format():
    """The expected-format hint is a string-only parser contract."""

    with pytest.raises(TypeError):
        parse_llm_json_output("{\"answer\": true}", expected_output_format={})


def test_parse_llm_json_output_does_not_use_expected_format_on_success(monkeypatch):
    """Valid object JSON should not call the LLM repair path."""

    def _unexpected_repair(_broken_string: str, *, expected_output_format=None):
        del expected_output_format
        raise AssertionError("LLM repair should not run for valid JSON objects")

    monkeypatch.setattr(utils_module, "parse_json_with_llm", _unexpected_repair)

    result = parse_llm_json_output(
        "{\"answer\": true}",
        expected_output_format='{"answer": true}',
    )

    assert result == {"answer": True}


def test_parse_json_with_llm_renders_expected_format_in_system_prompt(monkeypatch):
    """Expected output format belongs in the repair prompt, not user JSON."""

    captured_messages = {}

    class _RepairLLM:
        def invoke(self, messages, *, config):
            del config
            captured_messages["messages"] = messages
            return_value = type(
                "_Response",
                (),
                {"content": "{\"tool_calls\": []}"},
            )()
            return return_value

    expected_format = """{
  "tool_calls": []
}"""
    monkeypatch.setattr(utils_module, "_parse_json_with_llm", _RepairLLM())

    result = utils_module.parse_json_with_llm(
        "[]",
        expected_output_format=expected_format,
    )

    system_prompt, human_message = captured_messages["messages"]
    assert isinstance(system_prompt, SystemMessage)
    assert isinstance(human_message, HumanMessage)
    assert expected_format in system_prompt.content
    assert "Expected output format from the original prompt:" in system_prompt.content
    payload = json.loads(human_message.content)
    assert payload == {"broken_json": "[]"}
    assert "expected_output_format" not in human_message.content
    assert result == {"tool_calls": []}


def test_parse_json_with_llm_omits_expected_format_header_when_absent(monkeypatch):
    """The base repair prompt should not include an empty expected-format block."""

    captured_messages = {}

    class _RepairLLM:
        def invoke(self, messages, *, config):
            del config
            captured_messages["messages"] = messages
            return_value = type(
                "_Response",
                (),
                {"content": "{\"answer\": true}"},
            )()
            return return_value

    monkeypatch.setattr(utils_module, "_parse_json_with_llm", _RepairLLM())

    result = utils_module.parse_json_with_llm("{answer: true}")

    system_prompt = captured_messages["messages"][0]
    assert "Expected output format from the original prompt:" not in system_prompt.content
    assert result == {"answer": True}


@pytest.mark.live_llm
def test_parse_json_with_llm_live_wraps_empty_array_contract(
    ensure_json_repair_live_llm,
    monkeypatch,
):
    """The repair prompt should map a bare no-op array into the target object."""

    del ensure_json_repair_live_llm
    live_llm = _CapturingRepairLLM(utils_module._parse_json_with_llm)
    monkeypatch.setattr(utils_module, "_parse_json_with_llm", live_llm)

    broken_json = "[]"
    result = utils_module.parse_json_with_llm(
        broken_json,
        expected_output_format=_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT,
    )
    _write_json_repair_trace(
        "wraps_empty_array_contract",
        broken_json=broken_json,
        expected_output_format=_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT,
        raw_repair_output=live_llm.raw_output,
        parsed_output=result,
    )

    assert result == {"tool_calls": []}


@pytest.mark.live_llm
def test_parse_json_with_llm_live_wraps_tool_call_array_contract(
    ensure_json_repair_live_llm,
    monkeypatch,
):
    """The repair prompt should preserve tool-call entries while fixing wrapper shape."""

    del ensure_json_repair_live_llm
    live_llm = _CapturingRepairLLM(utils_module._parse_json_with_llm)
    monkeypatch.setattr(utils_module, "_parse_json_with_llm", live_llm)

    broken_json = """[
  {
    "tool": "send_message",
    "args": {
      "target_channel": "same",
      "text": "I will remind you now.",
      "execute_at": "2026-04-23 06:12"
    }
  }
]"""
    result = utils_module.parse_json_with_llm(
        broken_json,
        expected_output_format=_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT,
    )
    _write_json_repair_trace(
        "wraps_tool_call_array_contract",
        broken_json=broken_json,
        expected_output_format=_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT,
        raw_repair_output=live_llm.raw_output,
        parsed_output=result,
    )

    assert list(result) == ["tool_calls"]
    assert isinstance(result["tool_calls"], list)
    assert len(result["tool_calls"]) == 1
    tool_call = result["tool_calls"][0]
    assert tool_call["tool"] == "send_message"
    assert tool_call["args"]["target_channel"] == "same"
    assert tool_call["args"]["execute_at"] == "2026-04-23 06:12"


@pytest.mark.live_llm
def test_parse_json_with_llm_live_repairs_malformed_object_contract(
    ensure_json_repair_live_llm,
    monkeypatch,
):
    """The repair prompt should fix syntax without copying schema placeholders."""

    del ensure_json_repair_live_llm
    live_llm = _CapturingRepairLLM(utils_module._parse_json_with_llm)
    monkeypatch.setattr(utils_module, "_parse_json_with_llm", live_llm)

    broken_json = """```json
{
  tool_calls: [
    {
      tool: "send_message",
      args: {
        target_channel: "same",
        text: "I will remind you now.",
        execute_at: "2026-04-23 06:12",
      },
    },
  ],
}
```"""
    result = utils_module.parse_json_with_llm(
        broken_json,
        expected_output_format=_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT,
    )
    _write_json_repair_trace(
        "repairs_malformed_object_contract",
        broken_json=broken_json,
        expected_output_format=_TOOL_CALLS_EXPECTED_OUTPUT_FORMAT,
        raw_repair_output=live_llm.raw_output,
        parsed_output=result,
    )

    assert list(result) == ["tool_calls"]
    tool_call = result["tool_calls"][0]
    assert tool_call["tool"] == "send_message"
    assert tool_call["args"]["target_channel"] == "same"
    assert tool_call["args"]["text"] == "I will remind you now."


class TestBuildAffinityBlock:
    def test_hostile(self):
        result = build_affinity_block(100)
        assert result["level"] == "Scornful"
        assert "contempt" in result["instruction"] or "dismissive" in result["instruction"]

    def test_cold(self):
        result = build_affinity_block(300)
        assert result["level"] == "Reserved"
        assert "brief" in result["instruction"] or "professional" in result["instruction"]

    def test_neutral(self):
        result = build_affinity_block(AFFINITY_DEFAULT)
        assert result["level"] == "Neutral"

    def test_friendly(self):
        result = build_affinity_block(700)
        assert result["level"] == "Warm"
        assert "warmth" in result["instruction"] or "enthusiasm" in result["instruction"]

    def test_devoted(self):
        result = build_affinity_block(900)
        assert result["level"] == "Protective"
        assert "protective" in result["instruction"] or "loyalty" in result["instruction"]
