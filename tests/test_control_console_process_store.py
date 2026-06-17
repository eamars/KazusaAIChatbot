"""Local process-state store contract tests."""

from __future__ import annotations

import pytest


def test_state_store_recovers_desired_state_and_generation_atomically(tmp_path) -> None:
    """Restarted console processes should recover desired and owned PID state."""

    from control_console.process_store import ProcessStore

    store = ProcessStore(tmp_path)
    store.set_desired_state("brain", "running")
    store.record_process_owner(
        service_id="brain",
        pid=4242,
        generation="generation-1",
        command_fingerprint="fingerprint-1",
    )

    recovered = ProcessStore(tmp_path)
    snapshot = recovered.load_snapshot()

    assert snapshot["services"]["brain"]["desired_state"] == "running"
    assert snapshot["services"]["brain"]["pid"] == 4242
    assert snapshot["services"]["brain"]["generation"] == "generation-1"
    assert snapshot["services"]["brain"]["command_fingerprint"] == "fingerprint-1"
    assert not (tmp_path / "services.json.tmp").exists()


def test_state_store_reports_invalid_json_and_write_failures(
    tmp_path,
    monkeypatch,
) -> None:
    """Corrupt or unwritable process-state files should fail explicitly."""

    from control_console import process_store as process_store_module
    from control_console.process_store import ProcessStore

    store = ProcessStore(tmp_path)
    (tmp_path / "services.json").write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        store.load_snapshot()

    (tmp_path / "services.json").unlink()

    def fail_replace(source, destination) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(process_store_module.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="cannot write"):
        store.set_desired_state("brain", "running")
