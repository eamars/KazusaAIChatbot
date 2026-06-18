"""Local control-console store resilience tests."""

from __future__ import annotations


def test_process_log_tail_skips_malformed_jsonl_rows(tmp_path) -> None:
    """One corrupt process-log row should not break the whole log page."""

    from control_console.log_store import ProcessLogStore

    log_store = ProcessLogStore(tmp_path / "logs")
    log_store.append_line(
        service_id="brain",
        stream="supervisor",
        line="service started",
    )
    path = tmp_path / "logs" / "brain.jsonl"
    path.write_text(
        "{not-json}\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    lines = log_store.tail(service_id="brain", limit=10)

    assert len(lines) == 1
    assert lines[0].line == "service started"


def test_audit_read_recent_skips_malformed_jsonl_rows(tmp_path) -> None:
    """One corrupt audit row should not break bootstrap or audit pages."""

    from control_console.audit import LocalAuditWriter

    audit_writer = LocalAuditWriter(tmp_path / "audit.jsonl")
    audit_writer.write_event(
        event_type="service_started",
        operator_id="operator",
        service_id="brain",
    )
    path = tmp_path / "audit.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8") + "{not-json}\n",
        encoding="utf-8",
    )

    events = audit_writer.read_recent(limit=10)

    assert len(events) == 1
    assert events[0].event_type == "service_started"
