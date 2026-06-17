"""Local process-state store contract tests."""

from __future__ import annotations


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

