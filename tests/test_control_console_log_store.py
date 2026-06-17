"""Bounded process-log store tests."""

from __future__ import annotations


def test_log_store_returns_bounded_redacted_lines(tmp_path) -> None:
    """Log tails should be bounded and redacted before API use."""

    from control_console.log_store import ProcessLogStore

    store = ProcessLogStore(tmp_path)
    for index in range(5):
        store.append_line(
            service_id="brain",
            stream="stdout",
            line=f"line {index} api_key=secret",
        )

    result = store.tail(service_id="brain", limit=3)
    rendered = repr(result)

    assert len(result) == 3
    assert [line.line for line in result][0].startswith("line 2")
    assert "secret" not in rendered
    assert all(line.cursor for line in result)

