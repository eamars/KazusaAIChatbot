"""Deterministic contracts for persistent repository indexing."""

from pathlib import Path

import hashlib
import sqlite3

import pytest


def test_complete_snapshot_is_pinned_and_queryable_only_after_publish(
    tmp_path: Path,
) -> None:
    """Index every safe file and publish one complete immutable snapshot."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    for number in range(121):
        (source_root / f"module_{number}.py").write_text(
            f"VALUE_{number} = {number}\n",
            encoding="utf-8",
        )
    (source_root / "target.py").write_text(
        "DISCOVERABLE_TARGET = True\n",
        encoding="utf-8",
    )

    snapshot = build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )

    assert snapshot["status"] == "complete"
    results = search_snapshot(
        workspace_root=tmp_path / "workspace",
        snapshot_id=snapshot["snapshot_id"],
        mode="literal",
        query="DISCOVERABLE_TARGET",
    )
    assert [row["repo_path"] for row in results["rows"]] == ["target.py"]


def test_index_excludes_secret_like_content_in_a_large_file_middle(
    tmp_path: Path,
) -> None:
    """Exclude a file when a credential marker appears beyond its first chunk."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    secret_text = "x" * 9000 + "\nAPI_KEY=secret-value\n" + "x" * 9000
    (source_root / "safe_name.txt").write_text(secret_text, encoding="utf-8")

    snapshot = build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    results = search_snapshot(
        workspace_root=tmp_path / "workspace",
        snapshot_id=snapshot["snapshot_id"],
        mode="literal",
        query="secret-value",
    )

    assert results["rows"] == []
    assert snapshot["excluded_paths"] == ["safe_name.txt"]


def test_stale_cursor_is_rejected(tmp_path: Path) -> None:
    """Bind search cursors to their exact snapshot and normalized query."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "one.py").write_text("needle = 1\n", encoding="utf-8")
    snapshot = build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    first_page = search_snapshot(
        workspace_root=tmp_path / "workspace",
        snapshot_id=snapshot["snapshot_id"],
        mode="literal",
        query="needle",
    )
    stale_page = search_snapshot(
        workspace_root=tmp_path / "workspace",
        snapshot_id=snapshot["snapshot_id"],
        mode="literal",
        query="different",
        cursor=first_page["cursor"],
    )

    assert stale_page["status"] == "stale_cursor"


def test_regex_timeout_returns_no_partial_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Reject a timed-out regex search without exposing partial evidence."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index import search

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "target.py").write_text("needle = 1\n", encoding="utf-8")
    snapshot = build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    monkeypatch.setattr(search, "_regex_rows", lambda **_kwargs: None)

    result = search.search_snapshot(
        workspace_root=tmp_path / "workspace",
        snapshot_id=snapshot["snapshot_id"],
        mode="regex",
        query="needle",
    )

    assert result == {"status": "regex_timeout", "rows": [], "cursor": None}


def test_snapshot_uses_fts5_and_keeps_a_prior_complete_snapshot(
    tmp_path: Path,
) -> None:
    """Publish a new immutable snapshot without invalidating a pinned prior one."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "module.py"
    source_path.write_text("FIRST_UNIQUE_VALUE = 1\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    first_snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    source_path.write_text("SECOND_UNIQUE_VALUE = 2\n", encoding="utf-8")
    second_snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )

    assert first_snapshot["snapshot_id"] != second_snapshot["snapshot_id"]
    first_result = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=first_snapshot["snapshot_id"],
        mode="literal",
        query="FIRST_UNIQUE_VALUE",
    )
    assert [row["repo_path"] for row in first_result["rows"]] == ["module.py"]
    database_path = next((workspace_root / "repository_indexes").rglob(
        f"{second_snapshot['snapshot_id']}.sqlite"
    ))
    connection = sqlite3.connect(database_path)
    try:
        table_names = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert "chunk_fts" in table_names


def test_storage_reclaims_only_complete_unpinned_snapshots(tmp_path: Path) -> None:
    """Preserve a ledger-pinned snapshot while reclaiming an unpinned one."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.storage import (
        reclaim_unpinned_snapshots,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "module.py"
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    source_path.write_text("FIRST = 1\n", encoding="utf-8")
    first_snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    source_path.write_text("SECOND = 2\n", encoding="utf-8")
    second_snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )

    reclaimed = reclaim_unpinned_snapshots(
        workspace_root=workspace_root,
        source_identity=source_identity,
        ledgers=[
            {
                "status": "awaiting_approval",
                "index_snapshot_id": first_snapshot["snapshot_id"],
            }
        ],
        active_snapshot_ids=set(),
    )

    assert reclaimed == [second_snapshot["snapshot_id"]]
    index_directory = workspace_root / "repository_indexes"
    assert list(index_directory.rglob(f"{first_snapshot['snapshot_id']}.sqlite"))
    assert not list(index_directory.rglob(f"{second_snapshot['snapshot_id']}.sqlite"))


def test_snapshot_exposes_canonical_rows_and_all_search_modes(
    tmp_path: Path,
) -> None:
    """Index Python symbols and bind every search page to its candidate view."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "widget.py").write_text(
        "import helpers\n\n\ndef build_widget() -> str:\n    return 'needle'\n",
        encoding="utf-8",
    )
    workspace_root = tmp_path / "workspace"
    snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )

    database_path = next(
        (workspace_root / "repository_indexes").rglob(
            f"{snapshot['snapshot_id']}.sqlite"
        )
    )
    connection = sqlite3.connect(database_path)
    try:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()

    symbol_result = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        mode="symbol",
        query="build_widget",
        overlay_revision=3,
    )
    path_result = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        mode="path",
        query="widget.py",
        overlay_revision=3,
    )
    stale_result = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        mode="path",
        query="widget.py",
        overlay_revision=4,
        cursor=path_result["cursor"],
    )

    assert {"file", "chunk", "chunk_fts", "symbol", "import_edge"} <= table_names
    assert symbol_result["status"] == "ok"
    assert symbol_result["rows"][0]["symbol"] == "build_widget"
    assert path_result["rows"][0]["repo_path"] == "widget.py"
    assert path_result["rows"][0]["candidate_revision"] == 3
    assert stale_result["status"] == "stale_cursor"


def test_index_resource_limit_returns_a_typed_blocker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Report the exact exhausted index resource without dropping safe text."""

    from kazusa_ai_chatbot.coding_agent.repository_index import builder

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "large.py").write_text("x" * 64, encoding="utf-8")
    monkeypatch.setattr(builder, "MAX_INDEX_FILE_BYTES", 32)

    result = builder.build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )

    assert result == {
        "status": "blocked",
        "blocker_type": "index_resource_exhausted",
        "resource": "MAX_INDEX_FILE_BYTES",
    }


def test_incomplete_snapshot_resumes_and_is_never_queryable_before_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume retained build state rather than deleting it or exposing it."""

    from kazusa_ai_chatbot.coding_agent.repository_index import builder
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "alpha.py").write_text("ALPHA = 1\n", encoding="utf-8")
    (source_root / "bravo.py").write_text("BRAVO = 2\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    original_insert_rows = builder._insert_rows

    def interrupted_insert_rows(*args, **kwargs) -> None:
        original_insert_rows(*args, **kwargs)
        raise RuntimeError("simulated index interruption")

    monkeypatch.setattr(builder, "_insert_rows", interrupted_insert_rows)
    with pytest.raises(RuntimeError, match="simulated index interruption"):
        builder.build_repository_snapshot(
            workspace_root=workspace_root,
            source_root=source_root,
            source_identity=source_identity,
        )

    building_database = next(
        (workspace_root / "repository_indexes").rglob("*.building.sqlite")
    )
    provisional_snapshot_id = building_database.name.removesuffix(".building.sqlite")
    with pytest.raises(FileNotFoundError):
        search_snapshot(
            workspace_root=workspace_root,
            snapshot_id=provisional_snapshot_id,
            mode="literal",
            query="ALPHA",
        )

    monkeypatch.setattr(builder, "_insert_rows", original_insert_rows)
    resumed = builder.build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )

    assert resumed["status"] == "complete"
    assert resumed["resumed_incomplete_build"] is True
    assert not building_database.exists()
    assert search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=resumed["snapshot_id"],
        mode="literal",
        query="BRAVO",
    )["status"] == "ok"
    database_path = building_database.with_name(
        f"{resumed['snapshot_id']}.sqlite"
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM file").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM chunk").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone() == (
            2,
        )


def test_snapshot_pin_release_controls_reclamation(
    tmp_path: Path,
) -> None:
    """Retain a complete snapshot until its explicit durable pin is released."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.storage import (
        pin_snapshot,
        reclaim_released_snapshots,
        release_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "module.py"
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    source_path.write_text("FIRST = 1\n", encoding="utf-8")
    first = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    source_path.write_text("SECOND = 2\n", encoding="utf-8")
    second = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )

    pin_snapshot(
        workspace_root=workspace_root,
        source_identity=source_identity,
        snapshot_id=first["snapshot_id"],
        owner_id="run-one",
    )
    assert reclaim_released_snapshots(
        workspace_root=workspace_root,
        source_identity=source_identity,
    ) == [second["snapshot_id"]]
    release_snapshot(
        workspace_root=workspace_root,
        source_identity=source_identity,
        snapshot_id=first["snapshot_id"],
        owner_id="run-one",
    )
    assert reclaim_released_snapshots(
        workspace_root=workspace_root,
        source_identity=source_identity,
    ) == [first["snapshot_id"]]


def test_snapshot_reports_reused_unchanged_content_hashes(tmp_path: Path) -> None:
    """Reuse known-safe unchanged content during an incremental snapshot build."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    shared_content = "def stable_symbol() -> str:\n    return 'stable'\n"
    (source_root / "stable.py").write_text(shared_content, encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    (source_root / "new.py").write_text("NEW_VALUE = 1\n", encoding="utf-8")

    incremental = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )

    assert incremental["reused_content_hashes"] == [
        hashlib.sha256(shared_content.encode("utf-8")).hexdigest()
    ]


def test_snapshot_reuses_unchanged_content_after_path_rename(tmp_path: Path) -> None:
    """Reuse compatible indexed rows by content while assigning new path ids."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    content = "def stable_symbol() -> str:\n    return 'stable'\n"
    original = source_root / "original.py"
    original.write_text(content, encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    original.rename(source_root / "renamed.py")

    snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )

    assert snapshot["reused_content_hashes"] == [
        hashlib.sha256(content.encode("utf-8")).hexdigest()
    ]


def test_candidate_view_pagination_never_repeats_overlay_rows(
    tmp_path: Path,
) -> None:
    """Page one merged base/overlay order without tombstones or repetition."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.overlay import (
        CandidateOverlay,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        search_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    for index in range(30):
        (source_root / f"base_{index:02d}.py").write_text(
            "TOKEN = 'match'\n",
            encoding="utf-8",
        )
    workspace_root = tmp_path / "workspace"
    snapshot = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    overlay_path = tmp_path / "candidate" / "overlay.sqlite"
    overlay = CandidateOverlay(overlay_path)
    try:
        for index in range(15):
            overlay.delete(repo_path=f"base_{index:02d}.py", revision=1)
        for index in range(10):
            overlay.upsert(
                repo_path=f"overlay_{index:02d}.py",
                content="TOKEN = 'match'\n",
                revision=1,
            )
    finally:
        overlay.close()

    first = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=str(snapshot["snapshot_id"]),
        mode="literal",
        query="match",
        overlay_revision=1,
        overlay_database_path=overlay_path,
    )
    second = search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=str(snapshot["snapshot_id"]),
        mode="literal",
        query="match",
        cursor=str(first["cursor"]),
        overlay_revision=1,
        overlay_database_path=overlay_path,
    )
    paths = [row["repo_path"] for row in [*first["rows"], *second["rows"]]]

    assert len(paths) == 25
    assert len(paths) == len(set(paths))
    assert not {
        f"base_{index:02d}.py" for index in range(15)
    } & set(paths)
    assert second["cursor"] is None


def test_reclamation_keeps_active_and_building_snapshots(tmp_path: Path) -> None:
    """Never remove a cursor-pinned or incomplete snapshot during reclamation."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )
    from kazusa_ai_chatbot.coding_agent.repository_index.storage import (
        reclaim_released_snapshots,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "module.py"
    workspace_root = tmp_path / "workspace"
    source_identity = {"current_commit": "fixture"}
    source_path.write_text("FIRST = 1\n", encoding="utf-8")
    first = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    source_path.write_text("SECOND = 2\n", encoding="utf-8")
    second = build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity=source_identity,
    )
    index_directory = next((workspace_root / "repository_indexes").iterdir())
    building_path = index_directory / "incomplete.building.sqlite"
    building_path.write_bytes(b"incomplete")

    reclaimed = reclaim_released_snapshots(
        workspace_root=workspace_root,
        source_identity=source_identity,
        active_snapshot_ids={first["snapshot_id"]},
    )

    assert reclaimed == [second["snapshot_id"]]
    assert (index_directory / f"{first['snapshot_id']}.sqlite").is_file()
    assert building_path.is_file()


def test_large_safe_file_is_chunked_and_late_lines_remain_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stream bounded line-aware chunks without losing later read evidence."""

    from kazusa_ai_chatbot.coding_agent.repository_index import builder
    from kazusa_ai_chatbot.coding_agent.repository_index.search import (
        read_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    content = "".join(f"LINE_{number:03d} = {number}\n" for number in range(80))
    (source_root / "large.py").write_text(content, encoding="utf-8")
    monkeypatch.setattr(builder, "MAX_INDEX_CHUNK_BYTES", 128)
    workspace_root = tmp_path / "workspace"
    snapshot = builder.build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    database_path = next(
        (workspace_root / "repository_indexes").rglob(
            f"{snapshot['snapshot_id']}.sqlite",
        ),
    )
    connection = sqlite3.connect(database_path)
    try:
        maximum_chunk_bytes = connection.execute(
            "SELECT MAX(length(CAST(content AS BLOB))) FROM chunk",
        ).fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    finally:
        connection.close()

    late_read = read_snapshot(
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        repo_path="large.py",
        start_line=70,
        end_line=72,
    )

    assert chunk_count > 1
    assert maximum_chunk_bytes <= 128
    assert late_read["rows"][0]["content"] == (
        "LINE_069 = 69\nLINE_070 = 70\nLINE_071 = 71\n"
    )


def test_repository_identity_uses_canonical_nested_json() -> None:
    """Keep source identity stable across nested mapping insertion order."""

    from kazusa_ai_chatbot.coding_agent.repository_index.identity import (
        source_identity_hash,
    )

    first = {"repository": {"owner": "fixture", "name": "demo"}, "ref": "main"}
    second = {"ref": "main", "repository": {"name": "demo", "owner": "fixture"}}

    assert source_identity_hash(first) == source_identity_hash(second)


def test_secret_path_fragments_and_pem_content_are_excluded(tmp_path: Path) -> None:
    """Exclude credential-like basenames and standard private-key markers."""

    from kazusa_ai_chatbot.coding_agent.repository_index.builder import (
        build_repository_snapshot,
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "my-secret-config.txt").write_text(
        "ordinary text\n",
        encoding="utf-8",
    )
    (source_root / "innocent.txt").write_text(
        "prefix\n-----BEGIN PRIVATE KEY-----\nvalue\n",
        encoding="utf-8",
    )
    snapshot = build_repository_snapshot(
        workspace_root=tmp_path / "workspace",
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )

    assert snapshot["excluded_paths"] == [
        "innocent.txt",
        "my-secret-config.txt",
    ]


def test_regex_query_and_snapshot_storage_limits_are_typed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject oversized regex input and storage without silent truncation."""

    from kazusa_ai_chatbot.coding_agent.repository_index import builder, search

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    snapshot = builder.build_repository_snapshot(
        workspace_root=workspace_root,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )
    monkeypatch.setattr(search, "MAX_REGEX_QUERY_CHARS", 3)
    regex_result = search.search_snapshot(
        workspace_root=workspace_root,
        snapshot_id=snapshot["snapshot_id"],
        mode="regex",
        query="VALUE",
    )

    limited_workspace = tmp_path / "limited"
    monkeypatch.setattr(builder, "MAX_INDEX_STORAGE_BYTES", 4)
    storage_result = builder.build_repository_snapshot(
        workspace_root=limited_workspace,
        source_root=source_root,
        source_identity={"current_commit": "fixture"},
    )

    assert regex_result == {"status": "rejected", "rows": [], "cursor": None}
    assert storage_result == {
        "status": "blocked",
        "blocker_type": "index_resource_exhausted",
        "resource": "MAX_INDEX_STORAGE_BYTES",
    }
