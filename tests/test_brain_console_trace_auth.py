"""Brain-side authorization tests for Control Console trace disclosure."""

from __future__ import annotations

from starlette.requests import Request


def _request(headers: dict[str, str]) -> Request:
    """Build a minimal Starlette request with exact browser headers."""

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": [
            (name.lower().encode("ascii"), value.encode("ascii"))
            for name, value in headers.items()
        ],
    }
    return Request(scope)


def _chat_request(platform: str = "debug"):
    """Build a valid typed request for the trace disclosure helper."""

    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest

    return ChatRequest.model_validate({
        "platform": platform,
        "platform_channel_id": "debug-channel",
        "channel_type": "private",
        "platform_message_id": "debug-message",
        "platform_user_id": "operator",
        "message_envelope": {
            "body_text": "Look up the trace id.",
            "raw_wire_text": "Look up the trace id.",
            "mentions": [],
            "reply": None,
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
    })


def test_valid_console_headers_authorize_exact_trace_id(monkeypatch) -> None:
    """The configured shared secret authorizes only the Debug Chat surface."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "KAZUSA_CONTROL_BRAIN_SHARED_SECRET", "shared")
    request = _chat_request()
    service._authorize_console_trace_request(
        request=_request({
            "X-Kazusa-Control-Console": "debug-v1",
            "X-Kazusa-Control-Console-Auth": "shared",
        }),
        chat_request=request,
    )

    assert service._operator_trace_id(
        request=request,
        trace_id="llmtrace_console",
        trace_recorded=True,
    ) == "llmtrace_console"


def test_invalid_or_non_debug_headers_fail_closed(monkeypatch) -> None:
    """Missing, wrong, and non-debug authorization never disclose a trace."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "KAZUSA_CONTROL_BRAIN_SHARED_SECRET", "shared")
    cases = [
        ({}, "debug"),
        ({"X-Kazusa-Control-Console": "debug-v1"}, "debug"),
        ({
            "X-Kazusa-Control-Console": "debug-v1",
            "X-Kazusa-Control-Console-Auth": "wrong",
        }, "debug"),
        ({
            "X-Kazusa-Control-Console": "debug-v2",
            "X-Kazusa-Control-Console-Auth": "shared",
        }, "debug"),
        ({
            "X-Kazusa-Control-Console": "debug-v1",
            "X-Kazusa-Control-Console-Auth": "shared",
        }, "qq"),
    ]

    for headers, platform in cases:
        request = _chat_request(platform)
        service._authorize_console_trace_request(
            request=_request(headers),
            chat_request=request,
        )
        assert service._operator_trace_id(
            request=request,
            trace_id="llmtrace_hidden",
            trace_recorded=True,
        ) == ""


def test_unrecorded_trace_is_unavailable_even_when_authorized(monkeypatch) -> None:
    """Authorization cannot turn a skipped trace capture into an id."""

    from kazusa_ai_chatbot import service

    monkeypatch.setattr(service, "KAZUSA_CONTROL_BRAIN_SHARED_SECRET", "shared")
    request = _chat_request()
    service._authorize_console_trace_request(
        request=_request({
            "X-Kazusa-Control-Console": "debug-v1",
            "X-Kazusa-Control-Console-Auth": "shared",
        }),
        chat_request=request,
    )

    assert service._operator_trace_id(
        request=request,
        trace_id="llmtrace_not_recorded",
        trace_recorded=False,
    ) == ""
