import subprocess
from pathlib import Path
import urllib.request

import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching import run
from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    GitHubSource,
    parse_github_source,
)
from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone
from kazusa_ai_chatbot.coding_agent.code_fetching.managed_clone import (
    ManagedCloneError,
    build_managed_checkout_paths,
    can_reuse_managed_checkout,
    write_metadata,
)
from kazusa_ai_chatbot.coding_agent.tools.git import GitCommandError


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout.strip()
    return output


def _make_git_checkout(tmp_path: Path) -> Path:
    repo_root = tmp_path / "local_repo"
    repo_root.mkdir()
    _run_git(["init"], repo_root)
    _run_git(["config", "user.email", "test@example.com"], repo_root)
    _run_git(["config", "user.name", "Test User"], repo_root)
    source_file = repo_root / "reader.py"
    source_file.write_text("print('hello')\n", encoding="utf-8")
    _run_git(["add", "reader.py"], repo_root)
    _run_git(["commit", "-m", "initial"], repo_root)
    _run_git(
        [
            "remote",
            "add",
            "origin",
            "https://github.com/eamars/KazusaAIChatbot.git",
        ],
        repo_root,
    )
    return repo_root


def test_github_parser_handles_repo_tree_blob_and_raw_sources() -> None:
    repo_source = parse_github_source(
        "https://github.com/eamars/KazusaAIChatbot.git"
    )
    assert repo_source is not None
    assert repo_source.owner == "eamars"
    assert repo_source.repo == "KazusaAIChatbot"
    assert repo_source.source_kind == "repository"
    assert repo_source.requested_ref is None
    assert repo_source.repo_relative_path is None

    tree_source = parse_github_source(
        "https://github.com/eamars/KazusaAIChatbot/tree/main/src"
    )
    assert tree_source is not None
    assert tree_source.source_kind == "directory"
    assert tree_source.requested_ref == "main"
    assert tree_source.repo_relative_path == "src"

    blob_source = parse_github_source(
        "https://github.com/eamars/KazusaAIChatbot/blob/main/src/app.py"
    )
    assert blob_source is not None
    assert blob_source.source_kind == "file"
    assert blob_source.requested_ref == "main"
    assert blob_source.repo_relative_path == "src/app.py"

    raw_source = parse_github_source(
        "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"
    )
    assert raw_source is not None
    assert raw_source.owner == "github"
    assert raw_source.repo == "gitignore"
    assert raw_source.source_kind == "file"
    assert raw_source.requested_ref == "main"
    assert raw_source.repo_relative_path == "Python.gitignore"

    raw_refs_source = parse_github_source(
        "https://raw.githubusercontent.com/eamars/"
        "OpenTrickler-RP2040-Controller/refs/heads/main/src/charge_mode.cpp"
    )
    assert raw_refs_source is not None
    assert raw_refs_source.owner == "eamars"
    assert raw_refs_source.repo == "OpenTrickler-RP2040-Controller"
    assert raw_refs_source.source_kind == "file"
    assert raw_refs_source.requested_ref == "refs/heads/main"
    assert raw_refs_source.repo_relative_path == "src/charge_mode.cpp"


async def test_run_downloads_raw_github_file_without_clone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    source_url = (
        "https://raw.githubusercontent.com/eamars/"
        "OpenTrickler-RP2040-Controller/refs/heads/main/src/charge_mode.cpp"
    )
    downloaded = b"// Define PID terms\nfloat fine_trickler_integral = 0.0f;\n"

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def read(self) -> bytes:
            return downloaded

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: int,
    ) -> FakeResponse:
        assert request.full_url == source_url
        assert timeout == 60
        return FakeResponse()

    def fail_clone(source, workspace_root: str):
        raise AssertionError("raw GitHub file should not require git clone")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fail_clone,
    )

    result = await run(
        {
            "source_url": source_url,
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "succeeded"
    assert result["repository"] is not None
    assert result["repository"]["storage_kind"] == "managed_download"
    assert result["repository"]["managed_checkout"] is True
    assert result["repository"]["current_commit"].startswith("raw-sha256:")
    assert result["source_scope"] is not None
    assert result["source_scope"]["kind"] == "file"
    assert result["source_scope"]["requested_ref"] == "refs/heads/main"
    assert result["source_scope"]["repo_relative_path"] == "src/charge_mode.cpp"

    local_root = Path(result["repository"]["local_root"])
    downloaded_path = local_root / "src" / "charge_mode.cpp"
    assert downloaded_path.read_bytes() == downloaded


async def test_run_resolves_captured_question_url_through_source_intake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone
    from kazusa_ai_chatbot.coding_agent.code_fetching import source_intake
    from kazusa_ai_chatbot.coding_agent.code_fetching.source_intake import (
        SourceIntakeResult,
        SourceMention,
    )

    task_text = (
        '@杏山千纱 千纱千纱，能不能帮我分析一下这个项目'
        'https://github.com/sdyzjx/open-yachiyo，辛苦你了'
    )

    async def fake_run_source_intake(
        task_text_arg: str,
        *,
        retry_feedback: list[str] | None = None,
    ) -> SourceIntakeResult:
        assert task_text_arg == task_text
        assert retry_feedback is None
        result = SourceIntakeResult(
            task_source_mode='single_primary',
            source_mentions=(
                SourceMention(
                    raw_text='https://github.com/sdyzjx/open-yachiyo',
                    role='primary_code_source',
                    family_hint='github_repository',
                ),
            ),
        )
        return result

    def fake_ensure_managed_checkout(source, workspace_root: str):
        checkout_root = tmp_path / 'managed_checkout'
        checkout_root.mkdir()
        repository = {
            'provider': 'github',
            'owner': source.owner,
            'repo': source.repo,
            'source_url': source.source_url,
            'requested_ref': source.requested_ref,
            'resolved_ref': source.requested_ref or 'main',
            'current_commit': 'a' * 40,
            'default_branch': 'main',
            'local_root': str(checkout_root),
            'storage_kind': 'managed_clone',
            'managed_checkout': True,
            'workspace_root': workspace_root,
            'cache_key': 'github-sdyzjx-open-yachiyo-main',
            'dirty_state': 'clean',
        }
        return repository

    monkeypatch.setattr(
        source_intake,
        'run_source_intake',
        fake_run_source_intake,
    )
    monkeypatch.setattr(
        managed_clone,
        'ensure_managed_checkout',
        fake_ensure_managed_checkout,
    )

    result = await run(
        {
            'question': task_text,
            'workspace_root': str(tmp_path / 'coding_workspace'),
        }
    )

    assert result['status'] == 'succeeded'
    assert result['repository'] is not None
    assert result['repository']['owner'] == 'sdyzjx'
    assert result['repository']['repo'] == 'open-yachiyo'
    assert result['source_scope'] is not None
    assert result['source_scope']['kind'] == 'repository'


async def test_run_materializes_inline_source_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import source_intake
    from kazusa_ai_chatbot.coding_agent.code_fetching.source_intake import (
        SourceIntakeResult,
        SourceMention,
    )

    task_text = (
        'Please inspect this code:\n'
        '```python\n'
        'def normalize(value):\n'
        '    return value.strip().lower()\n'
        '```\n'
    )

    async def fake_run_source_intake(
        task_text_arg: str,
        *,
        retry_feedback: list[str] | None = None,
    ) -> SourceIntakeResult:
        assert task_text_arg == task_text
        assert retry_feedback is None
        result = SourceIntakeResult(
            task_source_mode='inline_bundle',
            source_mentions=(
                SourceMention(
                    raw_text='def normalize(value):',
                    role='primary_code_source',
                    family_hint='inline_code',
                    language_hint='python',
                    filename_hint='normalizer.py',
                ),
            ),
        )
        return result

    monkeypatch.setattr(
        source_intake,
        'run_source_intake',
        fake_run_source_intake,
    )

    result = await run(
        {
            'question': task_text,
            'workspace_root': str(tmp_path / 'coding_workspace'),
        }
    )

    assert result['status'] == 'succeeded'
    assert result['repository'] is not None
    assert result['repository']['provider'] == 'inline'
    assert result['repository']['owner'] == 'inline'
    assert result['repository']['storage_kind'] == 'managed_inline_bundle'
    assert result['repository']['current_commit'].startswith('inline-sha256:')
    assert result['source_scope'] is not None
    assert result['source_scope']['kind'] == 'file'
    assert result['source_scope']['repo_relative_path'] == 'normalizer.py'
    assert 'managed_inline:materialized:1' in result['trace_summary']

    local_root = Path(result['repository']['local_root'])
    materialized_file = local_root / 'normalizer.py'
    assert materialized_file.read_text(encoding='utf-8') == (
        'def normalize(value):\n'
        '    return value.strip().lower()\n'
    )
    assert (local_root / 'manifest.json').exists()


@pytest.mark.parametrize(
    "source_url",
    [
        "git@github.com:owner/repo.git",
        "https://token@github.com/owner/repo",
        "https://gitlab.com/owner/repo",
        "https://github.com/owner/repo/issues/1",
        "https://github.com/owner/repo/archive/refs/heads/main.zip",
        "https://github.com/owner/repo/blob/main/.git/config",
        "https://gist.github.com/owner/1234567890",
        "npm:react",
    ],
)
async def test_run_rejects_unsupported_source_classes(
    source_url: str,
    tmp_path: Path,
) -> None:
    result = await run(
        {
            "source_url": source_url,
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "rejected"
    assert result["repository"] is None
    assert result["source_scope"] is None
    assert result["message"]
    assert result["limitations"]


@pytest.mark.parametrize(
    ("source_url", "expected_kind"),
    [
        ("https://github.com/owner/repo/blob/main/missing.py", "file"),
        ("https://github.com/owner/repo/tree/main/missing_dir", "directory"),
    ],
)
async def test_run_needs_input_for_missing_scoped_github_paths(
    monkeypatch: pytest.MonkeyPatch,
    source_url: str,
    expected_kind: str,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    def fake_ensure_managed_checkout(source, workspace_root: str):
        checkout_root = tmp_path / "managed_checkout"
        checkout_root.mkdir()
        existing_file = checkout_root / "existing.py"
        existing_file.write_text("print('present')\n", encoding="utf-8")
        repository = {
            "provider": "github",
            "owner": source.owner,
            "repo": source.repo,
            "source_url": source.source_url,
            "requested_ref": source.requested_ref,
            "resolved_ref": source.requested_ref or "main",
            "current_commit": "a" * 40,
            "default_branch": "main",
            "local_root": str(checkout_root),
            "storage_kind": "managed_clone",
            "managed_checkout": True,
            "workspace_root": workspace_root,
            "cache_key": "github-owner-repo-main",
            "dirty_state": "clean",
        }
        return repository

    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fake_ensure_managed_checkout,
    )

    result = await run(
        {
            "source_url": source_url,
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "needs_user_input"
    assert result["repository"] is None
    assert result["source_scope"] is None
    assert expected_kind in result["message"]


async def test_run_returns_needs_user_input_without_source(
    tmp_path: Path,
) -> None:
    result = await run(
        {
            "question": "How does this project read images?",
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "needs_user_input"
    assert result["repository"] is None
    assert result["source_scope"] is None


async def test_run_rejects_multi_repository_compare_requests(
    tmp_path: Path,
) -> None:
    result = await run(
        {
            "question": (
                "Compare https://github.com/owner/first "
                "with https://github.com/owner/second"
            ),
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "rejected"
    assert result["limitations"]
    assert result["repository"] is None
    assert result["source_scope"] is None
    assert "Multiple-source code reading" in result["message"]


async def test_run_uses_most_specific_same_repo_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    def fake_ensure_managed_checkout(source, workspace_root: str):
        checkout_root = tmp_path / "managed_checkout"
        checkout_root.mkdir()
        scoped_file = checkout_root / "src" / "app.py"
        scoped_file.parent.mkdir()
        scoped_file.write_text("print('app')\n", encoding="utf-8")
        repository = {
            "provider": "github",
            "owner": source.owner,
            "repo": source.repo,
            "source_url": source.source_url,
            "requested_ref": source.requested_ref,
            "resolved_ref": source.requested_ref or "main",
            "current_commit": "a" * 40,
            "default_branch": "main",
            "local_root": str(checkout_root),
            "storage_kind": "managed_clone",
            "managed_checkout": True,
            "workspace_root": workspace_root,
            "cache_key": "github-owner-repo-main",
            "dirty_state": "clean",
        }
        return repository

    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fake_ensure_managed_checkout,
    )

    result = await run(
        {
            "question": (
                "Read https://github.com/owner/repo and especially "
                "https://github.com/owner/repo/blob/main/src/app.py"
            ),
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "succeeded"
    assert result["repository"] is not None
    assert result["source_scope"] is not None
    assert result["source_scope"]["kind"] == "file"
    assert result["source_scope"]["repo_relative_path"] == "src/app.py"


async def test_run_preserves_unsupported_limitations_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    def fake_ensure_managed_checkout(source, workspace_root: str):
        checkout_root = tmp_path / "managed_checkout"
        checkout_root.mkdir()
        repository = {
            "provider": "github",
            "owner": source.owner,
            "repo": source.repo,
            "source_url": source.source_url,
            "requested_ref": source.requested_ref,
            "resolved_ref": source.requested_ref or "main",
            "current_commit": "a" * 40,
            "default_branch": "main",
            "local_root": str(checkout_root),
            "storage_kind": "managed_clone",
            "managed_checkout": True,
            "workspace_root": workspace_root,
            "cache_key": "github-owner-repo-main",
            "dirty_state": "clean",
        }
        return repository

    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fake_ensure_managed_checkout,
    )

    result = await run(
        {
            "source_url": "https://github.com/owner/repo",
            "repo_url": "https://github.com/owner/repo/issues/1",
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "succeeded"
    assert result["repository"] is not None
    assert result["source_scope"] is not None
    assert result["limitations"]
    assert "issues" in result["limitations"][0]


async def test_run_sanitizes_managed_checkout_failure_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    sensitive_temp_root = tmp_path / "coding_workspace" / "t"

    def fake_ensure_managed_checkout(source, workspace_root: str):
        raise ManagedCloneError(
            "git command failed with exit code 128: "
            f"Cloning into '{sensitive_temp_root}' failed; "
            "raw git stderr; cache_key=github-owner-repo-main"
        )

    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fake_ensure_managed_checkout,
    )

    result = await run(
        {
            "source_url": "https://github.com/owner/repo",
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    rendered_result = repr(result)
    assert result["status"] == "failed"
    assert result["message"] == "Unable to prepare managed checkout."
    assert result["limitations"] == ["Managed checkout preparation failed."]
    assert str(tmp_path) not in rendered_result
    assert "raw git stderr" not in rendered_result
    assert "github-owner-repo-main" not in rendered_result


async def test_run_needs_input_for_managed_clone_source_access_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    def fake_ensure_managed_checkout(source, workspace_root: str):
        raise ManagedCloneError(
            "managed git clone failed: git command failed with exit code 128"
        )

    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fake_ensure_managed_checkout,
    )

    result = await run(
        {
            "source_url": "https://github.com/owner/repo",
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    rendered_result = repr(result)
    assert result["status"] == "needs_user_input"
    assert result["message"] == "Unable to access requested GitHub source."
    assert "exit code 128" not in rendered_result
    assert str(tmp_path) not in rendered_result


async def test_run_needs_input_for_raw_download_source_access_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_download

    source_url = "https://raw.githubusercontent.com/owner/repo/main/src/app.py"

    def fake_ensure_managed_raw_file_download(source, workspace_root: str):
        raise managed_download.ManagedDownloadError(
            "raw GitHub file download failed: HTTP Error 404"
        )

    monkeypatch.setattr(
        managed_download,
        "ensure_managed_raw_file_download",
        fake_ensure_managed_raw_file_download,
    )

    result = await run(
        {
            "source_url": source_url,
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    rendered_result = repr(result)
    assert result["status"] == "needs_user_input"
    assert result["message"] == "Unable to access requested raw GitHub file."
    assert "HTTP Error 404" not in rendered_result
    assert str(tmp_path) not in rendered_result


async def test_run_sanitizes_managed_raw_download_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_download

    workspace_root = tmp_path / "coding_workspace"
    source_url = "https://raw.githubusercontent.com/owner/repo/main/src/app.py"
    source = parse_github_source(source_url)
    assert source is not None
    paths = build_managed_checkout_paths(
        workspace_root=str(workspace_root),
        provider="github_raw_file",
        owner=source.owner,
        repo=source.repo,
        requested_ref=source.requested_ref,
        require_checkout_path_budget=False,
    )
    temporary_root = Path(paths["temporary_root"])
    temporary_root.mkdir(parents=True)

    def fake_download_raw_file(_source_url: str) -> bytes:
        return b"print('fixture')\n"

    def fail_cleanup(path: Path) -> None:
        raise PermissionError(f"locked cleanup path: {path}")

    monkeypatch.setattr(
        managed_download,
        "_download_raw_file",
        fake_download_raw_file,
    )
    monkeypatch.setattr(managed_download.shutil, "rmtree", fail_cleanup)

    result = await run(
        {
            "source_url": source_url,
            "workspace_root": str(workspace_root),
        }
    )

    rendered_result = repr(result)
    assert result["status"] == "failed"
    assert result["message"] == "Unable to prepare managed raw file download."
    assert result["limitations"] == ["Managed raw file download failed."]
    assert str(tmp_path) not in rendered_result
    assert "locked cleanup path" not in rendered_result


async def test_run_resolves_existing_local_checkout_without_mutation(
    tmp_path: Path,
) -> None:
    repo_root = _make_git_checkout(tmp_path)
    status_before = _run_git(["status", "--porcelain"], repo_root)

    result = await run(
        {
            "question": (
                '[eamars/KazusaAIChatbot]'
                '(https://github.com/eamars/KazusaAIChatbot) '
                '项目是怎么实现读图的'
            ),
            "local_root_hint": str(repo_root),
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    status_after = _run_git(["status", "--porcelain"], repo_root)
    assert status_after == status_before
    assert result["status"] == "succeeded"
    assert result["repository"] is not None
    assert result["repository"]["storage_kind"] == "existing_local_checkout"
    assert result["repository"]["managed_checkout"] is False
    assert result["repository"]["owner"] == "eamars"
    assert result["repository"]["repo"] == "KazusaAIChatbot"
    assert result["source_scope"] is not None
    assert result["source_scope"]["kind"] == "repository"
    assert (
        result["source_scope"]["source_url"]
        == "local://github/eamars/KazusaAIChatbot"
    )


async def test_run_resolves_local_path_as_file_scope(tmp_path: Path) -> None:
    repo_root = _make_git_checkout(tmp_path)
    local_file = repo_root / "reader.py"

    result = await run(
        {
            "local_path_hint": str(local_file),
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "succeeded"
    assert result["repository"] is not None
    assert result["source_scope"] is not None
    assert result["source_scope"]["kind"] == "file"
    assert result["source_scope"]["repo_relative_path"] == "reader.py"
    assert (
        result["source_scope"]["source_url"]
        == "local://github/eamars/KazusaAIChatbot/reader.py"
    )


async def test_run_rejects_local_git_internal_path(tmp_path: Path) -> None:
    repo_root = _make_git_checkout(tmp_path)
    git_config_path = repo_root / ".git" / "config"

    result = await run(
        {
            "local_path_hint": str(git_config_path),
            "workspace_root": str(tmp_path / "coding_workspace"),
        }
    )

    assert result["status"] == "rejected"
    assert result["repository"] is None
    assert result["source_scope"] is None


def test_managed_workspace_paths_and_metadata_guard(tmp_path: Path) -> None:
    paths = build_managed_checkout_paths(
        workspace_root=str(tmp_path / "coding_workspace"),
        provider="github",
        owner="owner",
        repo="repo",
        requested_ref="feature/test",
    )

    workspace_root = tmp_path / "coding_workspace"
    assert Path(paths["checkout_root"]).is_relative_to(workspace_root)
    assert Path(paths["metadata_path"]).is_relative_to(workspace_root)
    assert Path(paths["temporary_root"]).is_relative_to(workspace_root)
    assert Path(paths["lock_path"]).is_relative_to(workspace_root)

    metadata = {
        "schema_version": 1,
        "provider": "github",
        "owner": "owner",
        "repo": "repo",
        "source_url": "https://github.com/owner/repo",
        "requested_ref": "feature/test",
        "resolved_ref": "feature/test",
        "current_commit": "a" * 40,
    }
    write_metadata(paths["metadata_path"], metadata)

    assert can_reuse_managed_checkout(paths["metadata_path"], metadata)

    mismatch = dict(metadata)
    mismatch["repo"] = "other"
    assert not can_reuse_managed_checkout(paths["metadata_path"], mismatch)


def test_managed_checkout_paths_use_compact_hash_storage(
    tmp_path: Path,
) -> None:
    paths = build_managed_checkout_paths(
        workspace_root=str(tmp_path / "coding_workspace"),
        provider="github",
        owner="very-long-owner-name",
        repo="very-long-repository-name",
        requested_ref="feature/very-long-branch-name",
    )

    workspace_root = tmp_path / "coding_workspace"
    forbidden_parts = {
        "very-long-owner-name",
        "very-long-repository-name",
        "feature_very-long-branch-name",
    }
    for key in ("checkout_root", "metadata_path", "temporary_root", "lock_path"):
        relative_parts = Path(paths[key]).relative_to(workspace_root).parts
        assert not forbidden_parts.intersection(relative_parts)
        assert len(relative_parts) <= 4

    other_paths = build_managed_checkout_paths(
        workspace_root=str(tmp_path / "coding_workspace"),
        provider="github",
        owner="very-long-owner-name",
        repo="other-repository-name",
        requested_ref="feature/very-long-branch-name",
    )
    assert paths["temporary_root"] != other_paths["temporary_root"]


def test_managed_checkout_rejects_workspace_root_with_no_path_budget(
    tmp_path: Path,
) -> None:
    deep_workspace = tmp_path
    for index in range(20):
        deep_workspace = deep_workspace / f"long-segment-{index:02d}"

    with pytest.raises(ManagedCloneError, match="workspace root"):
        build_managed_checkout_paths(
            workspace_root=str(deep_workspace),
            provider="github",
            owner="owner",
            repo="repo",
            requested_ref=None,
        )


def test_managed_checkout_accepts_normal_deep_workspace_budget() -> None:
    deep_workspace = (
        Path("test_artifacts")
        / "coding_agent_live_workspaces"
        / "phase1_hard_gates"
        / "normal_deep_workspace_root"
        / "managed_checkout_area"
        / "case_with_long_but_supported_workspace"
    )

    paths = build_managed_checkout_paths(
        workspace_root=str(deep_workspace),
        provider="github",
        owner="owner",
        repo="repo",
        requested_ref=None,
    )

    checkout_root = Path(paths["checkout_root"])
    assert checkout_root.is_relative_to(deep_workspace.resolve(strict=False))
    assert len(str(checkout_root)) <= 220


def test_managed_clone_fetches_requested_commit_after_clone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_ref = "a" * 40
    source = GitHubSource(
        owner="owner",
        repo="repo",
        source_url="https://github.com/owner/repo",
        source_kind="repository",
        requested_ref=requested_ref,
        repo_relative_path=None,
    )
    paths = build_managed_checkout_paths(
        workspace_root=str(tmp_path / "coding_workspace"),
        provider="github",
        owner=source.owner,
        repo=source.repo,
        requested_ref=source.requested_ref,
    )
    temporary_root = Path(paths["temporary_root"])
    checkout_root = Path(paths["checkout_root"])
    calls: list[tuple[list[str], str | None]] = []

    def fake_git(args: list[str], *, cwd: str | None = None) -> object:
        calls.append((args, cwd))
        if args[:3] == ["-c", "core.longpaths=true", "clone"]:
            assert "--branch" not in args
            temporary_root.mkdir(parents=True)
            return object()
        if args == [
            "-c",
            "core.longpaths=true",
            "fetch",
            "--depth",
            "1",
            "origin",
            requested_ref,
        ]:
            assert cwd == str(temporary_root)
            return object()
        if args == [
            "-c",
            "core.longpaths=true",
            "checkout",
            "--detach",
            "FETCH_HEAD",
        ]:
            assert cwd == str(temporary_root)
            return object()
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(managed_clone, "run_git_command", fake_git)

    managed_clone._clone_into_managed_checkout(source, paths)

    assert checkout_root.exists()
    assert not temporary_root.exists()
    assert calls[1][0] == [
        "-c",
        "core.longpaths=true",
        "fetch",
        "--depth",
        "1",
        "origin",
        requested_ref,
    ]
    assert calls[2][0] == [
        "-c",
        "core.longpaths=true",
        "checkout",
        "--detach",
        "FETCH_HEAD",
    ]


def test_managed_clone_preserves_git_failure_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = GitHubSource(
        owner="owner",
        repo="repo",
        source_url="https://github.com/owner/repo",
        source_kind="repository",
        requested_ref=None,
        repo_relative_path=None,
    )
    paths = build_managed_checkout_paths(
        workspace_root=str(tmp_path / "coding_workspace"),
        provider="github",
        owner=source.owner,
        repo=source.repo,
        requested_ref=source.requested_ref,
    )
    temporary_root = Path(paths["temporary_root"])

    def fail_git_clone(args: list[str]) -> object:
        assert args[:3] == ["-c", "core.longpaths=true", "clone"]
        temporary_root.mkdir(parents=True)
        raise GitCommandError("primary clone failure")

    def fail_cleanup(path: Path, **_kwargs) -> None:
        raise PermissionError("cleanup target is locked")

    monkeypatch.setattr(managed_clone, "run_git_command", fail_git_clone)
    monkeypatch.setattr(managed_clone.shutil, "rmtree", fail_cleanup)

    with pytest.raises(ManagedCloneError, match="primary clone failure"):
        managed_clone._clone_into_managed_checkout(source, paths)
