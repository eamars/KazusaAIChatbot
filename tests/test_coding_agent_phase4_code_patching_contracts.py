from pathlib import Path


def test_code_patching_rejects_ambiguous_existing_file_anchor(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_path = repo_root / "app.py"
    target_path.write_text(
        "VALUE = 1\nVALUE = 1\n",
        encoding="utf-8",
    )

    patch_artifacts, _, changed_files, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "replace-value",
                "kind": "replace",
                "path": "app.py",
                "anchor": "VALUE = 1\n",
                "content": "VALUE = 2\n",
                "summary": "Replace one value.",
            }
        ],
        max_files=4,
        max_diff_chars=4000,
    )

    assert patch_artifacts == []
    assert changed_files == []
    assert any("multiple" in error for error in errors)


def test_code_patching_rejects_mixed_package_atomically(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_path = repo_root / "app.py"
    target_path.write_text("VALUE = 1\n", encoding="utf-8")

    patch_artifacts, created_files, changed_files, errors = (
        compile_patch_operations(
            repo_root=repo_root,
            patch_operations=[
                {
                    "operation_id": "create-helper",
                    "kind": "create_file",
                    "path": "helper.py",
                    "content": "HELPER = True\n",
                    "summary": "Create helper.",
                },
                {
                    "operation_id": "bad-replace",
                    "kind": "replace",
                    "path": "app.py",
                    "anchor": "MISSING = 1\n",
                    "content": "VALUE = 2\n",
                    "summary": "Replace missing anchor.",
                },
            ],
            max_files=4,
            max_diff_chars=4000,
        )
    )

    assert patch_artifacts == []
    assert created_files == []
    assert changed_files == []
    assert any("not found" in error for error in errors)


def test_code_patching_rejects_create_file_over_empty_existing_file(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_path = repo_root / "empty.py"
    target_path.write_text("", encoding="utf-8")

    patch_artifacts, created_files, changed_files, errors = (
        compile_patch_operations(
            repo_root=repo_root,
            patch_operations=[
                {
                    "operation_id": "create-empty",
                    "kind": "create_file",
                    "path": "empty.py",
                    "content": "VALUE = 1\n",
                    "summary": "Create file.",
                }
            ],
            max_files=4,
            max_diff_chars=4000,
        )
    )

    assert patch_artifacts == []
    assert created_files == []
    assert changed_files == []
    assert any("existing file" in error for error in errors)


def test_code_patching_enforces_small_full_file_replacement_cap(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        REPLACE_FILE_SMALL_MAX_CHARS,
        compile_patch_operations,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_path = repo_root / "small.py"
    target_path.write_text("VALUE = 1\n", encoding="utf-8")
    oversized_content = "A" * (REPLACE_FILE_SMALL_MAX_CHARS + 1)

    patch_artifacts, _, _, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "replace-file",
                "kind": "replace_file_small",
                "path": "small.py",
                "content": oversized_content,
                "summary": "Replace whole file.",
            }
        ],
        max_files=4,
        max_diff_chars=REPLACE_FILE_SMALL_MAX_CHARS * 2,
    )

    assert patch_artifacts == []
    assert any("full-file" in error for error in errors)


def test_code_patching_materializes_review_package_under_patch_root(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
        VALIDATION_ROOT_NAME,
        materialize_patch_artifacts_for_review,
    )

    repo_root = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    repo_root.mkdir()
    workspace_root.mkdir()
    target_path = repo_root / "app.py"
    target_path.write_text("VALUE = 1\n", encoding="utf-8")

    patch_artifacts, _, _, errors = compile_patch_operations(
        repo_root=repo_root,
        patch_operations=[
            {
                "operation_id": "replace-value",
                "kind": "replace",
                "path": "app.py",
                "anchor": "VALUE = 1\n",
                "content": "VALUE = 2\n",
                "summary": "Replace one value.",
            }
        ],
        max_files=4,
        max_diff_chars=4000,
    )

    assert errors == []
    validation = materialize_patch_artifacts_for_review(
        repo_root=repo_root,
        workspace_root=workspace_root,
        patch_artifacts=patch_artifacts,
        max_files=4,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["sandbox_applied"] is True
    assert (workspace_root / VALIDATION_ROOT_NAME).is_dir()


def test_code_patching_allows_importable_test_dependency_in_source_free_package(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
        compile_patch_operations,
    )
    from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
        materialize_patch_artifacts_for_review,
    )

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    patch_artifacts, _, _, errors = compile_patch_operations(
        repo_root=None,
        patch_operations=[
            {
                "operation_id": "create-source",
                "kind": "create_file",
                "path": "src/csv_normalizer_cli.py",
                "content": (
                    "def normalize(text: str) -> str:\n"
                    "    return text.strip()\n"
                ),
                "summary": "Create CSV normalizer source.",
            },
            {
                "operation_id": "create-tests",
                "kind": "create_file",
                "path": "tests/test_csv_normalizer_tests.py",
                "content": (
                    "import pytest\n\n"
                    "from csv_normalizer_cli import normalize\n\n\n"
                    "def test_normalize_trims_text() -> None:\n"
                    "    assert normalize(' value ') == 'value'\n\n\n"
                    "def test_normalize_requires_text() -> None:\n"
                    "    with pytest.raises(AttributeError):\n"
                    "        normalize(None)\n"
                ),
                "summary": "Create CSV normalizer tests.",
            },
        ],
        max_files=4,
        max_diff_chars=4000,
    )

    assert errors == []
    validation = materialize_patch_artifacts_for_review(
        repo_root=None,
        workspace_root=workspace_root,
        patch_artifacts=patch_artifacts,
        max_files=4,
        max_diff_chars=4000,
    )

    assert validation["status"] == "succeeded"
    assert validation["errors"] == []
