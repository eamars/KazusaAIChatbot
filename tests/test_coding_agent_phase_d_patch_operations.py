"""Contracts for delete and rename patch operations."""

from pathlib import Path


def test_delete_and_rename_records_bind_digest_review_and_apply(
    tmp_path: Path,
) -> None:
    """Compile guarded delete and rename operations into reviewable diffs."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "old.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "remove.py").write_text("REMOVE = 1\n", encoding="utf-8")

    artifacts, _, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {"kind": "rename_file", "path": "old.py", "target_path": "new.py"},
            {"kind": "delete_file", "path": "remove.py"},
        ],
        max_files=4,
        max_diff_chars=4000,
    )

    assert errors == []
    assert {row["path"] for row in changed_files} == {
        "old.py",
        "new.py",
        "remove.py",
    }
    assert "rename from old.py" in "\n".join(
        artifact["diff_text"] for artifact in artifacts
    )
    assert "+++ /dev/null" in "\n".join(
        artifact["diff_text"] for artifact in artifacts
    )


def test_canonical_delete_and_rename_records_bind_digest_and_preconditions(
    tmp_path: Path,
) -> None:
    """Bind reviewable delete/rename records to source hashes and revision."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        build_canonical_operation_records,
        canonical_proposal_digest,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "old.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "remove.py").write_text("REMOVE = 1\n", encoding="utf-8")

    records = build_canonical_operation_records(
        repo_root=repo_root,
        patch_operations=[
            {"kind": "rename_file", "path": "old.py", "target_path": "new.py"},
            {"kind": "delete_file", "path": "remove.py"},
        ],
        candidate_revision=3,
    )

    assert records[0]["kind"] == "rename_file"
    assert records[0]["source_path"] == "old.py"
    assert records[0]["target_path"] == "new.py"
    assert records[0]["expected_candidate_revision"] == 3
    assert records[0]["expected_source_sha256"] == records[0]["content_sha256"]
    assert records[1]["kind"] == "delete_file"
    assert records[1]["target_path"] is None
    assert len(canonical_proposal_digest(records)) == 64


def test_canonical_digest_validation_rejects_missing_or_mismatched_records(
    tmp_path: Path,
) -> None:
    """Keep approved apply bound to the exact reviewed operation sequence."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        canonical_proposal_digest,
        validate_canonical_operation_binding,
    )

    records = [{
        "operation_id": "delete-one",
        "kind": "delete_file",
        "source_path": "old.py",
        "target_path": None,
        "expected_source_sha256": "a" * 64,
        "expected_candidate_revision": 2,
        "result_sha256": None,
        "content_sha256": "a" * 64,
    }]
    digest = canonical_proposal_digest(records)

    assert validate_canonical_operation_binding(
        records=records,
        proposal_digest=digest,
        candidate_revision=3,
    ) == ""
    assert "digest" in validate_canonical_operation_binding(
        records=records,
        proposal_digest="b" * 64,
        candidate_revision=3,
    )
    assert "revision" in validate_canonical_operation_binding(
        records=records,
        proposal_digest=digest,
        candidate_revision=2,
    )


def test_approved_apply_rejects_mismatched_canonical_binding() -> None:
    """Reject canonical provenance drift before any apply authorization path."""

    from kazusa_ai_chatbot.coding_agent.code_patching.apply import (
        materialize_managed_candidate,
    )

    response = materialize_managed_candidate({
        "canonical_operation_records": [{
            "operation_id": "delete-one",
            "kind": "delete_file",
            "source_path": "old.py",
            "target_path": None,
            "expected_source_sha256": "a" * 64,
            "expected_candidate_revision": 2,
            "result_sha256": None,
            "content_sha256": "a" * 64,
        }],
        "proposal_digest": "b" * 64,
        "candidate_revision": 3,
    })

    assert response["status"] == "rejected"
    assert "digest" in response["limitations"][0].casefold()


def test_approved_apply_returns_validated_canonical_provenance(
    tmp_path: Path,
) -> None:
    """Echo the reviewed canonical delete record after managed application."""

    from kazusa_ai_chatbot.coding_agent.code_patching.apply import (
        apply_approved_patch,
    )
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        build_canonical_operation_records,
        canonical_proposal_digest,
        compile_patch_operations,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "remove.py").write_text("REMOVE = 1\n", encoding="utf-8")
    operations = [{"kind": "delete_file", "path": "remove.py"}]
    artifacts, _, _, errors = compile_patch_operations(
        repo_root=source_root,
        patch_operations=operations,
        max_files=4,
        max_diff_chars=4000,
    )
    assert errors == []
    records = build_canonical_operation_records(
        repo_root=source_root,
        patch_operations=operations,
        candidate_revision=0,
    )
    source_identity = {
        "provider": "github",
        "owner": "fixture",
        "repo": "demo",
        "current_commit": "abc123",
        "dirty_state": "clean",
    }
    response = apply_approved_patch({
        "workspace_root": str(tmp_path / "workspace"),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": artifacts,
        "approval": {
            "approved": True,
            "approved_by": "contract-test",
            "approved_at": "2026-07-11T00:00:00Z",
            "approval_reason": "Focused deterministic test.",
        },
        "max_files": 4,
        "max_diff_chars": 4000,
        "canonical_operation_records": records,
        "proposal_digest": canonical_proposal_digest(records),
        "candidate_revision": 1,
    })

    assert response["status"] == "succeeded"
    assert response["canonical_operation_records"] == records
    assert response["proposal_digest"] == canonical_proposal_digest(records)


def test_source_free_create_then_replace_uses_candidate_overlay() -> None:
    """Treat a candidate-created path as the source-free edit baseline."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[
            {
                "kind": "create_file",
                "path": "app.py",
                "content": "VALUE = 1\n",
            },
            {
                "kind": "replace_file_small",
                "path": "app.py",
                "content": "VALUE = 2\n",
            },
        ],
        max_files=2,
        max_diff_chars=4000,
    )

    assert errors == []
    assert [row["path"] for row in created_files] == ["app.py"]
    assert [row["path"] for row in changed_files] == ["app.py"]
    assert "+VALUE = 2" in artifacts[0]["diff_text"]
    assert "VALUE = 1" not in artifacts[0]["diff_text"]


def test_source_free_create_then_delete_has_no_net_artifact() -> None:
    """Retire a candidate-created path without requiring a repository."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[
            {
                "kind": "create_file",
                "path": "temporary.py",
                "content": "VALUE = 1\n",
            },
            {"kind": "delete_file", "path": "temporary.py"},
        ],
        max_files=2,
        max_diff_chars=4000,
    )

    assert errors == []
    assert artifacts == []
    assert created_files == []
    assert changed_files == []


def test_source_free_create_rename_replace_preserves_provenance() -> None:
    """Transfer created-path provenance through rename and later replacement."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    artifacts, created_files, changed_files, errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[
            {
                "kind": "create_file",
                "path": "draft.py",
                "content": "VALUE = 1\n",
            },
            {
                "kind": "rename_file",
                "path": "draft.py",
                "target_path": "app.py",
            },
            {
                "kind": "replace_file_small",
                "path": "app.py",
                "content": "VALUE = 2\n",
            },
        ],
        max_files=2,
        max_diff_chars=4000,
    )

    assert errors == []
    assert [row["path"] for row in created_files] == ["app.py"]
    assert [row["path"] for row in changed_files] == ["app.py"]
    assert artifacts[0]["files"] == ["app.py"]
    assert "+VALUE = 2" in artifacts[0]["diff_text"]


def test_source_free_unknown_or_tombstoned_sources_are_rejected() -> None:
    """Keep absent source-free paths outside the candidate edit authority."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    _, _, _, unknown_errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[{
            "kind": "replace_file_small",
            "path": "unknown.py",
            "content": "VALUE = 2\n",
        }],
        max_files=2,
        max_diff_chars=4000,
    )
    _, _, _, tombstone_errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[
            {
                "kind": "create_file",
                "path": "app.py",
                "content": "VALUE = 1\n",
            },
            {"kind": "delete_file", "path": "app.py"},
            {
                "kind": "replace_file_small",
                "path": "app.py",
                "content": "VALUE = 2\n",
            },
        ],
        max_files=2,
        max_diff_chars=4000,
    )

    assert any("repository context" in error for error in unknown_errors)
    assert any("missing file" in error for error in tombstone_errors)


def test_source_free_rename_rejects_candidate_collision() -> None:
    """Reject rename targets already present in the candidate overlay."""

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    artifacts, _, _, errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[
            {"kind": "create_file", "path": "a.py", "content": "A = 1\n"},
            {"kind": "create_file", "path": "b.py", "content": "B = 1\n"},
            {
                "kind": "rename_file",
                "path": "a.py",
                "target_path": "b.py",
            },
        ],
        max_files=2,
        max_diff_chars=4000,
    )

    assert artifacts == []
    assert any("invalid source or target" in error for error in errors)


def test_source_free_records_reject_hash_and_revision_mismatch() -> None:
    """Keep overlay sequences bound to exact source hashes and revisions."""

    import hashlib

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        build_canonical_operation_records,
        canonical_proposal_digest,
        validate_canonical_operation_binding,
    )

    created = "VALUE = 1\n"
    operations = [
        {
            "kind": "create_file",
            "path": "app.py",
            "content": created,
            "expected_candidate_revision": 0,
        },
        {
            "kind": "replace_file_small",
            "path": "app.py",
            "content": "VALUE = 2\n",
            "expected_source_sha256": "f" * 64,
            "expected_candidate_revision": 1,
        },
    ]

    try:
        build_canonical_operation_records(
            repo_root=None,
            patch_operations=operations,
            candidate_revision=0,
        )
    except ValueError as exc:
        assert "hash is stale" in str(exc)
    else:
        raise AssertionError("stale source-free hash was accepted")

    operations[1]["expected_source_sha256"] = hashlib.sha256(
        created.encode("utf-8"),
    ).hexdigest()
    operations[1]["expected_candidate_revision"] = 3
    records = build_canonical_operation_records(
        repo_root=None,
        patch_operations=operations,
        candidate_revision=0,
    )
    assert "revision" in validate_canonical_operation_binding(
        records=records,
        proposal_digest=canonical_proposal_digest(records),
        candidate_revision=2,
    )


def test_existing_rename_then_edit_compiles_in_candidate_order(tmp_path) -> None:
    """Render a content-preserving rename before a target-path text edit."""

    import hashlib

    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        build_canonical_operation_records,
        compile_patch_operations,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    original = "VALUE = 1\n"
    (source_root / "old.py").write_text(original, encoding="utf-8")
    source_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
    operations = [
        {
            "operation_id": "rename-one",
            "kind": "rename_file",
            "path": "old.py",
            "target_path": "new.py",
            "expected_source_sha256": source_hash,
            "expected_candidate_revision": 0,
        },
        {
            "operation_id": "edit-two",
            "kind": "replace",
            "path": "new.py",
            "anchor": "VALUE = 1\n",
            "content": "VALUE = 2\n",
            "expected_source_sha256": source_hash,
            "expected_candidate_revision": 1,
        },
    ]

    artifacts, _, _, errors = compile_patch_operations(
        repo_root=source_root,
        patch_operations=operations,
        max_files=2,
        max_diff_chars=4000,
    )
    records = build_canonical_operation_records(
        repo_root=source_root,
        patch_operations=operations,
        candidate_revision=2,
    )

    assert errors == []
    assert "rename from old.py" in artifacts[0]["diff_text"]
    assert "diff --git a/new.py b/new.py" in artifacts[1]["diff_text"]
    assert records[1]["source_path"] == "new.py"
