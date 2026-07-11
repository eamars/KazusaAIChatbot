"""Contracts for action-loop candidate mutation recovery."""

import hashlib
from pathlib import Path

import pytest


def test_committed_operation_replay_requires_the_same_identity(
    tmp_path: Path,
) -> None:
    """Reject an operation id replay that carries different mutation data."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    candidate.apply_journaled_mutation(
        operation_id="create-one",
        kind="create_file",
        repo_path="module.py",
        replacement="VALUE = 1\n",
        expected_revision=0,
        expected_source_sha256=None,
    )

    with pytest.raises(ValueError, match="replay identity mismatch"):
        candidate.apply_journaled_mutation(
            operation_id="create-one",
            kind="create_file",
            repo_path="module.py",
            replacement="VALUE = 2\n",
            expected_revision=0,
            expected_source_sha256=None,
        )


def test_overlay_hides_tombstone_and_exposes_rename_target(tmp_path: Path) -> None:
    """Present one candidate view after a content-preserving rename."""

    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    overlay = CandidateOverlay(tmp_path / "overlay.sqlite")
    overlay.rename(
        source_path="old.py",
        target_path="new.py",
        content="VALUE = 1\n",
        revision=1,
    )

    rows = overlay.search("VALUE")

    assert [row["repo_path"] for row in rows] == ["new.py"]
    assert overlay.is_tombstoned("old.py") is True


def test_recovery_completes_or_rolls_back_one_journaled_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete a candidate-written operation without replaying its edit text."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    candidate = CandidateState.create(tmp_path / "candidate")

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "upsert", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="one",
            kind="create_file",
            repo_path="new.py",
            replacement="VALUE = 1\n",
            expected_revision=0,
            expected_source_sha256=None,
        )
    monkeypatch.undo()
    recovered = CandidateState.load(tmp_path / "candidate")
    recovered.recover()

    assert recovered.revision == 1
    assert (tmp_path / "candidate" / "source" / "new.py").read_text(
        encoding="utf-8",
    ) == "VALUE = 1\n"


def test_recovery_rolls_back_a_mismatched_candidate_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a corrupted prepared mutation instead of committing stale content."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    candidate = CandidateState.create(tmp_path / "candidate")

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "upsert", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="one",
            kind="create_file",
            repo_path="new.py",
            replacement="EXPECTED = 1\n",
            expected_revision=0,
            expected_source_sha256=None,
        )
    (candidate.root / "source" / "new.py").write_text(
        "CORRUPTED = 1\n",
        encoding="utf-8",
    )
    monkeypatch.undo()

    recovered = CandidateState.load(tmp_path / "candidate")

    with pytest.raises(ValueError, match="candidate recovery failed"):
        recovered.recover()
    assert recovered.revision == 0
    assert not (tmp_path / "candidate" / "source" / "new.py").exists()
    assert recovered.journal[0]["state"] == "rolled_back"


def test_overlay_searches_created_paths_and_python_symbols(tmp_path: Path) -> None:
    """Expose current candidate-only files through path and symbol search."""

    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    overlay = CandidateOverlay(tmp_path / "overlay.sqlite")
    overlay.upsert(
        repo_path="created_widget.py",
        content="def build_widget() -> str:\n    return 'ready'\n",
        revision=2,
    )

    path_rows = overlay.search("created_widget", mode="path")
    symbol_rows = overlay.search("build_widget", mode="symbol")

    assert path_rows[0]["repo_path"] == "created_widget.py"
    assert path_rows[0]["candidate_revision"] == 2
    assert symbol_rows[0]["symbol"] == "build_widget"
    assert symbol_rows[0]["start_line"] == 1
    overlay.close()


def test_recovery_commits_a_candidate_rename_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a rename from durable candidate state without replaying text."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    source_root = tmp_path / "candidate" / "source"
    (source_root / "old.py").write_text("VALUE = 1\n", encoding="utf-8")
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "rename", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="one",
            kind="rename_file",
            repo_path="old.py",
            target_path="new.py",
            replacement=None,
            expected_revision=0,
            expected_source_sha256=hashlib.sha256(
                b"VALUE = 1\n",
            ).hexdigest(),
        )
    monkeypatch.undo()

    recovered = CandidateState.load(tmp_path / "candidate")
    recovered.recover()

    assert recovered.revision == 1
    assert not (source_root / "old.py").exists()
    assert (source_root / "new.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert recovered.journal[0]["state"] == "committed"


def test_recovery_commits_a_candidate_delete_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a deletion from durable state without reapplying user intent."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    source_root = tmp_path / "candidate" / "source"
    (source_root / "obsolete.py").write_text("VALUE = 1\n", encoding="utf-8")
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "delete", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="one",
            kind="delete_file",
            repo_path="obsolete.py",
            replacement=None,
            expected_revision=0,
            expected_source_sha256=hashlib.sha256(
                b"VALUE = 1\n",
            ).hexdigest(),
        )
    monkeypatch.undo()

    recovered = CandidateState.load(tmp_path / "candidate")
    recovered.recover()

    assert recovered.revision == 1
    assert not (source_root / "obsolete.py").exists()
    assert recovered.journal[0]["state"] == "committed"


def test_journaled_mutation_commits_candidate_and_overlay_as_one_operation(
    tmp_path: Path,
) -> None:
    """Require the full four-phase journal without raw-content backup storage."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    operation = candidate.apply_journaled_mutation(
        operation_id="operation-one",
        kind="replace_file_small",
        repo_path="module.py",
        replacement="VALUE = 2\n",
        expected_revision=0,
        expected_source_sha256=None,
    )

    assert operation["state"] == "committed"
    assert operation["resulting_candidate_revision"] == 1
    assert "previous_content" not in operation
    assert (tmp_path / "candidate" / "overlay.sqlite").is_file()


def test_recovery_blocks_next_mutation_until_journal_identity_reconciles(
    tmp_path: Path,
) -> None:
    """Do not dispatch a later action while a prior journal row is unresolved."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import (
        CandidateState,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    _write_phase_state(candidate, {"operation_id": "one", "state": "prepared"})

    recovered = CandidateState.load(tmp_path / "candidate")
    with pytest.raises(ValueError, match="recovery"):
        recovered.require_recovered_before_next_action()


def test_recovery_commits_overlay_written_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advance only the ledger phase after candidate and overlay are durable."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState

    candidate = CandidateState.create(tmp_path / "candidate")

    def fail_commit(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated commit interruption")

    monkeypatch.setattr(CandidateState, "_commit_operation", fail_commit)
    with pytest.raises(RuntimeError, match="commit interruption"):
        candidate.apply_journaled_mutation(
            operation_id="one",
            kind="create_file",
            repo_path="a.py",
            replacement="VALUE = 1\n",
            expected_revision=0,
            expected_source_sha256=None,
        )
    monkeypatch.undo()
    recovered = CandidateState.load(candidate.root)
    recovered.recover()
    recovered.recover()
    assert recovered.revision == 1
    assert recovered.journal[0]["state"] == "committed"


def test_edit_recovery_safely_restores_candidate_and_content_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Roll back a corrupted edit from managed backup and overlay metadata."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    source_path = candidate.root / "source" / "module.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    overlay = CandidateOverlay(candidate.root / "overlay.sqlite")
    overlay.upsert(repo_path="module.py", content="VALUE = 1\n", revision=0)
    overlay.close()

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "upsert", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="edit-one",
            kind="replace_file_small",
            repo_path="module.py",
            replacement="VALUE = 2\n",
            expected_revision=0,
            expected_source_sha256=hashlib.sha256(
                b"VALUE = 1\n",
            ).hexdigest(),
        )

    source_path.write_text("CORRUPTED = 1\n", encoding="utf-8")
    monkeypatch.undo()
    recovered = CandidateState.load(candidate.root)
    with pytest.raises(ValueError, match="candidate recovery failed"):
        recovered.recover()

    assert source_path.read_text(encoding="utf-8") == "VALUE = 1\n"
    overlay = CandidateOverlay(candidate.root / "overlay.sqlite")
    assert overlay.describe_paths(["module.py"]) == [{
        "repo_path": "module.py",
        "state": "content",
        "content_sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        "revision": 0,
    }]
    overlay.close()
    operation = recovered.journal[0]
    assert operation["state"] == "rolled_back"
    assert "previous_content" not in operation
    assert "backup_sha256" not in operation


def test_delete_recovery_safely_restores_candidate_and_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore a deleted path when its interrupted candidate state diverges."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    source_path = candidate.root / "source" / "obsolete.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "delete", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="delete-one",
            kind="delete_file",
            repo_path="obsolete.py",
            replacement=None,
            expected_revision=0,
            expected_source_sha256=hashlib.sha256(
                b"VALUE = 1\n",
            ).hexdigest(),
        )

    source_path.write_text("UNEXPECTED = 1\n", encoding="utf-8")
    monkeypatch.undo()
    recovered = CandidateState.load(candidate.root)
    with pytest.raises(ValueError, match="candidate recovery failed"):
        recovered.recover()

    assert source_path.read_text(encoding="utf-8") == "VALUE = 1\n"
    overlay = CandidateOverlay(candidate.root / "overlay.sqlite")
    assert overlay.describe_paths(["obsolete.py"])[0]["state"] == "absent"
    overlay.close()
    assert recovered.journal[0]["state"] == "rolled_back"


def test_rename_recovery_safely_restores_both_paths_and_prior_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore both rename paths and their distinct prior overlay states."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    source_root = candidate.root / "source"
    source_path = source_root / "old.py"
    target_path = source_root / "new.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    overlay = CandidateOverlay(candidate.root / "overlay.sqlite")
    overlay.upsert(repo_path="old.py", content="VALUE = 1\n", revision=2)
    overlay.delete(repo_path="new.py", revision=2)
    overlay.close()

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "rename", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="rename-one",
            kind="rename_file",
            repo_path="old.py",
            target_path="new.py",
            replacement=None,
            expected_revision=0,
            expected_source_sha256=hashlib.sha256(
                b"VALUE = 1\n",
            ).hexdigest(),
        )

    target_path.write_text("CORRUPTED = 1\n", encoding="utf-8")
    monkeypatch.undo()
    recovered = CandidateState.load(candidate.root)
    with pytest.raises(ValueError, match="candidate recovery failed"):
        recovered.recover()

    assert source_path.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not target_path.exists()
    overlay = CandidateOverlay(candidate.root / "overlay.sqlite")
    assert overlay.describe_paths(["old.py", "new.py"]) == [
        {
            "repo_path": "old.py",
            "state": "content",
            "content_sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
            "revision": 2,
        },
        {
            "repo_path": "new.py",
            "state": "tombstone",
            "content_sha256": None,
            "revision": 2,
        },
    ]
    overlay.close()
    assert recovered.journal[0]["state"] == "rolled_back"


def test_recovery_blocks_without_mutation_when_backup_identity_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave current state untouched when rollback evidence is irreconcilable."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.state import CandidateState
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )

    candidate = CandidateState.create(tmp_path / "candidate")
    source_path = candidate.root / "source" / "module.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    def fail_overlay_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated overlay interruption")

    monkeypatch.setattr(CandidateOverlay, "upsert", fail_overlay_write)
    with pytest.raises(RuntimeError, match="overlay interruption"):
        candidate.apply_journaled_mutation(
            operation_id="edit-one",
            kind="replace_file_small",
            repo_path="module.py",
            replacement="VALUE = 2\n",
            expected_revision=0,
            expected_source_sha256=hashlib.sha256(
                b"VALUE = 1\n",
            ).hexdigest(),
        )

    operation = candidate.journal[0]
    before_path = operation["before_paths"][0]
    backup_path = candidate.root / before_path["backup_relative_path"]
    backup_path.unlink()
    source_path.write_text("CORRUPTED = 1\n", encoding="utf-8")
    monkeypatch.undo()
    recovered = CandidateState.load(candidate.root)
    with pytest.raises(ValueError, match="backup identity"):
        recovered.recover()

    assert source_path.read_text(encoding="utf-8") == "CORRUPTED = 1\n"
    assert recovered.journal[0]["state"] == "candidate_written"


def _write_phase_state(candidate, operation: dict[str, object]) -> None:
    """Arrange one authoritative crash phase without production helper APIs."""

    candidate.journal = [operation]
    candidate._save()
