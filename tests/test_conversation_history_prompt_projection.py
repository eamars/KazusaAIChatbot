"""Focused tests for the universal conversation-history LLM projection helper.

These tests verify:
- timestamp rendering (UTC storage, pre-formatted, absent)
- speaker resolution (display_name, name fallback, assistant fallback, unknown)
- body text resolution (body_text -> content -> text)
- reply_to rendering
- attachment image-block projection
- max_rows slicing
- missing / malformed fields
- no internal id leakage in output lines
"""


from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    *,
    display_name: str = "",
    role: str = "user",
    body_text: str = "",
    content: str = "",
    text: str = "",
    timestamp: str = "",
    reply_context: dict | None = None,
    reply_to_display_name: str = "",
    attachments: list | None = None,
    platform_user_id: str = "",
    global_user_id: str = "",
    name: str = "",
    broadcast: bool = False,
    message_id: str = "",
) -> dict:
    """Build a minimal conversation-history row for testing."""

    row: dict = {"role": role}
    if display_name:
        row["display_name"] = display_name
    if name:
        row["name"] = name
    if body_text:
        row["body_text"] = body_text
    if content:
        row["content"] = content
    if text:
        row["text"] = text
    if timestamp:
        row["timestamp"] = timestamp
    if reply_context is not None:
        row["reply_context"] = reply_context
    if reply_to_display_name:
        row["reply_to_display_name"] = reply_to_display_name
    if attachments is not None:
        row["attachments"] = attachments
    if platform_user_id:
        row["platform_user_id"] = platform_user_id
    if global_user_id:
        row["global_user_id"] = global_user_id
    if broadcast:
        row["broadcast"] = broadcast
    if message_id:
        row["message_id"] = message_id
    return row


# ---------------------------------------------------------------------------
# Speaker resolution
# ---------------------------------------------------------------------------

class TestSpeakerResolution:

    def test_display_name_used_as_speaker(self) -> None:
        rows = [_row(display_name="Alice", body_text="hello")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["Alice: hello"]

    def test_name_fallback_when_display_name_missing(self) -> None:
        rows = [_row(name="Bob", body_text="hi")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["Bob: hi"]

    def test_assistant_fallback_uses_character_name(self) -> None:
        rows = [_row(role="assistant", body_text="hey")]
        lines = project_conversation_history_for_llm(
            rows, character_name="Kazusa",
        )
        assert lines == ["Kazusa: hey"]

    def test_unknown_fallback_when_no_name_and_not_assistant(self) -> None:
        rows = [_row(role="user", body_text="anon")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["unknown: anon"]

    def test_display_name_whitespace_stripped(self) -> None:
        rows = [_row(display_name="  Alice  ", body_text="msg")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["Alice: msg"]


# ---------------------------------------------------------------------------
# Body text resolution
# ---------------------------------------------------------------------------

class TestBodyTextResolution:

    def test_body_text_preferred(self) -> None:
        rows = [_row(display_name="A", body_text="primary", content="secondary", text="tertiary")]
        lines = project_conversation_history_for_llm(rows)
        assert "primary" in lines[0]

    def test_content_fallback(self) -> None:
        rows = [_row(display_name="A", content="fallback_content")]
        lines = project_conversation_history_for_llm(rows)
        assert "fallback_content" in lines[0]

    def test_text_fallback(self) -> None:
        rows = [_row(display_name="A", text="fallback_text")]
        lines = project_conversation_history_for_llm(rows)
        assert "fallback_text" in lines[0]

    def test_empty_body_produces_empty_after_colon(self) -> None:
        rows = [_row(display_name="A")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["A: "]


# ---------------------------------------------------------------------------
# Timestamp rendering
# ---------------------------------------------------------------------------

class TestTimestampRendering:

    def test_preformatted_timestamp(self) -> None:
        rows = [_row(display_name="A", body_text="hi", timestamp="2026-06-13 15:04")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["[2026-06-13 15:04] A: hi"]

    def test_absent_timestamp_no_brackets(self) -> None:
        rows = [_row(display_name="A", body_text="hi")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["A: hi"]

    def test_whitespace_only_timestamp_treated_as_absent(self) -> None:
        rows = [_row(display_name="A", body_text="hi", timestamp="   ")]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["A: hi"]


# ---------------------------------------------------------------------------
# Reply rendering
# ---------------------------------------------------------------------------

class TestReplyRendering:

    def test_reply_context_display_name(self) -> None:
        rows = [_row(
            display_name="Bob",
            body_text="agree",
            reply_context={"reply_to_display_name": "Alice"},
        )]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["Bob reply_to Alice: agree"]

    def test_top_level_reply_to_display_name_fallback(self) -> None:
        rows = [_row(
            display_name="Bob",
            body_text="agree",
            reply_to_display_name="Alice",
        )]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["Bob reply_to Alice: agree"]

    def test_reply_context_takes_precedence_over_top_level(self) -> None:
        rows = [_row(
            display_name="Bob",
            body_text="agree",
            reply_context={"reply_to_display_name": "ContextAlice"},
            reply_to_display_name="TopAlice",
        )]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["Bob reply_to ContextAlice: agree"]

    def test_no_reply_when_absent(self) -> None:
        rows = [_row(display_name="Bob", body_text="msg")]
        lines = project_conversation_history_for_llm(rows)
        assert "reply_to" not in lines[0]

    def test_reply_with_timestamp(self) -> None:
        rows = [_row(
            display_name="Bob",
            body_text="agree",
            timestamp="2026-06-13 15:14",
            reply_context={"reply_to_display_name": "Alice"},
        )]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["[2026-06-13 15:14] Bob reply_to Alice: agree"]


# ---------------------------------------------------------------------------
# max_rows slicing
# ---------------------------------------------------------------------------

class TestMaxRows:

    def test_max_rows_keeps_last_n(self) -> None:
        rows = [
            _row(display_name="A", body_text="first"),
            _row(display_name="B", body_text="second"),
            _row(display_name="C", body_text="third"),
        ]
        lines = project_conversation_history_for_llm(rows, max_rows=2)
        assert len(lines) == 2
        assert "second" in lines[0]
        assert "third" in lines[1]

    def test_max_rows_none_returns_all(self) -> None:
        rows = [
            _row(display_name="A", body_text="first"),
            _row(display_name="B", body_text="second"),
        ]
        lines = project_conversation_history_for_llm(rows)
        assert len(lines) == 2

    def test_max_rows_larger_than_input(self) -> None:
        rows = [_row(display_name="A", body_text="only")]
        lines = project_conversation_history_for_llm(rows, max_rows=10)
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# No id leakage
# ---------------------------------------------------------------------------

class TestNoIdLeakage:

    def test_platform_user_id_not_in_output(self) -> None:
        rows = [_row(
            display_name="A",
            body_text="hi",
            platform_user_id="uid_12345",
        )]
        lines = project_conversation_history_for_llm(rows)
        assert "uid_12345" not in lines[0]
        assert "platform_user_id" not in lines[0]

    def test_global_user_id_not_in_output(self) -> None:
        rows = [_row(
            display_name="A",
            body_text="hi",
            global_user_id="guid_abcdef",
        )]
        lines = project_conversation_history_for_llm(rows)
        assert "guid_abcdef" not in lines[0]
        assert "global_user_id" not in lines[0]

    def test_broadcast_not_in_output(self) -> None:
        rows = [_row(
            display_name="A",
            body_text="hi",
            broadcast=True,
        )]
        lines = project_conversation_history_for_llm(rows)
        assert "broadcast" not in lines[0]

    def test_message_id_not_in_output(self) -> None:
        rows = [_row(
            display_name="A",
            body_text="hi",
            message_id="msg_99999",
        )]
        lines = project_conversation_history_for_llm(rows)
        assert "msg_99999" not in lines[0]
        assert "message_id" not in lines[0]


# ---------------------------------------------------------------------------
# Missing / malformed fields
# ---------------------------------------------------------------------------

class TestMalformedInput:

    def test_non_dict_row_skipped(self) -> None:
        rows = [
            "not a dict",
            _row(display_name="A", body_text="ok"),
        ]
        lines = project_conversation_history_for_llm(rows)
        assert len(lines) == 1
        assert "ok" in lines[0]

    def test_empty_rows(self) -> None:
        lines = project_conversation_history_for_llm([])
        assert lines == []

    def test_non_string_body_text_treated_as_empty(self) -> None:
        rows = [{"role": "user", "display_name": "A", "body_text": 123}]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["A: "]

    def test_non_string_display_name_falls_through(self) -> None:
        rows = [{"role": "user", "display_name": 42, "body_text": "msg"}]
        lines = project_conversation_history_for_llm(rows)
        assert lines == ["unknown: msg"]


# ---------------------------------------------------------------------------
# Attachment image-block projection
# ---------------------------------------------------------------------------

class TestAttachmentProjection:

    def test_image_attachment_appended(self) -> None:
        rows = [_row(
            display_name="A",
            body_text="look",
            attachments=[{
                "media_type": "image/png",
                "description": "a cat photo",
            }],
        )]
        lines = project_conversation_history_for_llm(rows)
        assert len(lines) == 1
        assert "look" in lines[0]
        assert "a cat photo" in lines[0]


# ---------------------------------------------------------------------------
# Combined / integration
# ---------------------------------------------------------------------------

class TestCombined:

    def test_full_group_chat_scenario(self) -> None:
        rows = [
            _row(
                display_name="蚝爹油",
                body_text="捡垃圾不是乐趣么",
                timestamp="2026-06-13 15:04:19",
                platform_user_id="673225019",
                global_user_id="256e8a10-c406-47e9-ac8f-efd270d18160",
            ),
            _row(
                display_name="1816",
                body_text="@杏山千纱 那蚝爹油跟你啥关系",
                timestamp="2026-06-13 15:14:43",
                reply_context={"reply_to_display_name": "杏山千纱"},
            ),
        ]
        lines = project_conversation_history_for_llm(rows)
        assert lines == [
            "[2026-06-13 15:04:19] 蚝爹油: 捡垃圾不是乐趣么",
            "[2026-06-13 15:14:43] 1816 reply_to 杏山千纱: @杏山千纱 那蚝爹油跟你啥关系",
        ]

    def test_assistant_row_with_character_name(self) -> None:
        rows = [
            _row(
                role="assistant",
                body_text="我来啦",
                timestamp="2026-06-13 15:06",
                broadcast=True,
            ),
        ]
        lines = project_conversation_history_for_llm(
            rows, character_name="杏山千纱",
        )
        assert lines == ["[2026-06-13 15:06] 杏山千纱: 我来啦"]
        assert "broadcast" not in lines[0]
