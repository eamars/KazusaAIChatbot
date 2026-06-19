"""Bounded process-log store tests."""

from __future__ import annotations

import pytest


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


def test_log_store_reports_missing_and_read_write_failures(
    tmp_path,
    monkeypatch,
) -> None:
    """Log store should make absent logs empty and IO failures explicit."""

    from pathlib import Path

    from control_console.log_store import ProcessLogStore

    store = ProcessLogStore(tmp_path)
    assert store.tail(service_id="brain", limit=10) == []

    original_read_text = Path.read_text
    original_write_text = Path.write_text

    def fail_read_text(self, *args, **kwargs):
        if self.name == "brain.jsonl":
            raise OSError("read failed")
        return original_read_text(self, *args, **kwargs)

    def fail_write_text(self, *args, **kwargs):
        if self.name == "brain.jsonl":
            raise OSError("write failed")
        return original_write_text(self, *args, **kwargs)

    store.append_line(service_id="brain", stream="stdout", line="hello")
    monkeypatch.setattr(Path, "read_text", fail_read_text)
    with pytest.raises(RuntimeError, match="cannot read"):
        store.tail(service_id="brain", limit=10)

    monkeypatch.setattr(Path, "read_text", original_read_text)
    monkeypatch.setattr(Path, "write_text", fail_write_text)
    with pytest.raises(RuntimeError, match="cannot write"):
        store.append_line(service_id="brain", stream="stdout", line="hello")


def test_log_store_publisher_failure_does_not_break_append(
    tmp_path,
    caplog,
) -> None:
    """Live-stream fanout failures should not break persisted log writes."""

    from control_console.log_store import ProcessLogStore

    def fail_publish(_log_line):
        raise RuntimeError("publisher boom")

    caplog.set_level("WARNING")
    store = ProcessLogStore(tmp_path, log_publisher=fail_publish)

    line = store.append_line(
        service_id="brain",
        stream="stdout",
        line="persist this",
    )

    assert line.line == "persist this"
    assert store.tail(service_id="brain", limit=10)[0].line == "persist this"
    assert "process log publisher failed: publisher boom" in caplog.text
