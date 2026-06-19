"""Local audit writer tests."""

from __future__ import annotations

import pytest


def test_audit_writer_reads_recent_and_reports_io_failures(
    tmp_path,
    monkeypatch,
) -> None:
    """Audit writer should bound reads and surface local IO failures."""

    from pathlib import Path

    from control_console.audit import LocalAuditWriter

    audit_path = tmp_path / "audit.jsonl"
    writer = LocalAuditWriter(audit_path)
    assert writer.read_recent(limit=5) == []

    writer.write_event(event_type="one", operator_id="operator")
    writer.write_event(event_type="two", operator_id="operator")
    assert [event.event_type for event in writer.read_recent(limit=1)] == ["two"]

    original_open = Path.open
    original_read_text = Path.read_text

    def fail_open(self, *args, **kwargs):
        if self.name == "audit.jsonl":
            raise OSError("write failed")
        return original_open(self, *args, **kwargs)

    def fail_read_text(self, *args, **kwargs):
        if self.name == "audit.jsonl":
            raise OSError("read failed")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(RuntimeError, match="cannot write"):
        writer.write_event(event_type="three", operator_id="operator")

    monkeypatch.setattr(Path, "open", original_open)
    monkeypatch.setattr(Path, "read_text", fail_read_text)
    with pytest.raises(RuntimeError, match="cannot read"):
        writer.read_recent(limit=5)
