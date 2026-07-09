from release_feed.cache import should_refresh


def test_should_refresh_uses_caller_timeout() -> None:
    assert should_refresh(
        cached_at_seconds=100,
        now_seconds=111,
        timeout_seconds=10,
    )


def test_should_not_refresh_before_timeout() -> None:
    assert not should_refresh(
        cached_at_seconds=100,
        now_seconds=109,
        timeout_seconds=10,
    )
