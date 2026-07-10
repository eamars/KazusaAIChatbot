"""Deterministic contracts for process-local scoped session media."""

from kazusa_ai_chatbot.media_inspection.session_cache import (
    clear_session_media,
    get_session_media,
    put_session_media,
)


def test_session_media_cache_is_scoped_and_returns_payload() -> None:
    """A cache reference cannot cross the platform/user scope boundary."""

    scope = ("debug", "channel-1", "user-1")
    clear_session_media(scope)
    refs = put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": "aGVsbG8=",
        "source_summary": "current upload",
    }])

    assert len(refs) == 1
    assert get_session_media(scope, refs[0]["cache_ref"]) is not None
    assert get_session_media(
        ("debug", "channel-1", "another-user"),
        refs[0]["cache_ref"],
    ) is None
