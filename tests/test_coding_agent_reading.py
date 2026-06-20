from pathlib import Path
from typing import Any

import pytest


def _make_repository(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "reading_repo"
    (repo_root / "src" / "orders").mkdir(parents=True)
    (repo_root / "docs").mkdir()
    (repo_root / "assets").mkdir()

    (repo_root / "README.md").write_text(
        "# Reading Fixture\nThe order service exposes source-only behavior.\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "orders" / "service.py").write_text(
        "\n".join(
            [
                "from orders.gateway import PaymentGateway",
                "",
                "class OrderService:",
                "    def submit_order(self, payload: dict) -> dict:",
                "        payment = PaymentGateway().charge(payload)",
                "        return {'status': payment['status']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "orders" / "gateway.py").write_text(
        "\n".join(
            [
                "class PaymentGateway:",
                "    def charge(self, payload: dict) -> dict:",
                "        return {'status': 'charged', 'id': payload['id']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / ".env").write_text(
        "SECRET_TOKEN=do-not-read\n",
        encoding="utf-8",
    )
    (repo_root / "assets" / "logo.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00fixture"
    )

    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "reading-repo",
        "source_url": "https://github.com/fixture/reading-repo",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "b" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "github-fixture-reading-repo-main",
        "dirty_state": "clean",
    }
    return repository


def _scope(
    kind: str = "repository",
    repo_relative_path: str | None = None,
) -> dict[str, Any]:
    scope = {
        "kind": kind,
        "repo_relative_path": repo_relative_path,
        "source_url": "local://fixture/reading-repo",
        "requested_ref": None,
        "interpretation": "fixture scope",
    }
    return scope


def _run_reading(
    tmp_path: Path,
    *,
    question: str,
    source_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    result = run(
        {
            "question": question,
            "repository": _make_repository(tmp_path),
            "source_scope": source_scope or _scope(),
            "preferred_language": "English",
            "max_answer_chars": 1600,
        }
    )
    return result


def _assert_public_shape(result: dict[str, Any]) -> None:
    assert result["status"] in {
        "succeeded",
        "failed",
        "needs_user_input",
        "rejected",
    }
    assert isinstance(result["answer_text"], str)
    assert isinstance(result["evidence"], list)
    assert isinstance(result["limitations"], list)
    assert isinstance(result["trace_summary"], list)
    assert "reading_repo" not in repr(result)
    assert "workspace" not in repr(result)
    assert "github-fixture-reading-repo-main" not in repr(result)
    assert "SECRET_TOKEN" not in repr(result)


def test_missing_question_returns_needs_user_input(tmp_path: Path) -> None:
    result = _run_reading(tmp_path, question="")

    _assert_public_shape(result)
    assert result["status"] == "needs_user_input"
    assert result["evidence"] == []


@pytest.mark.parametrize(
    ("question", "expected_fragment"),
    [
        ("Apply a patch to rewrite OrderService.", "read-only"),
        ("Run pytest before answering.", "execute"),
        ("Inspect .env and tell me SECRET_TOKEN.", "environment files"),
        ("Dump the full raw contents of src/orders/service.py.", "raw file"),
        ("Analyze binary pixels in assets/logo.png.", "binary assets"),
    ],
)
def test_rejects_unsupported_phase1_requests(
    tmp_path: Path,
    question: str,
    expected_fragment: str,
) -> None:
    result = _run_reading(tmp_path, question=question)

    _assert_public_shape(result)
    assert result["status"] == "rejected"
    assert expected_fragment in result["answer_text"]
    assert result["evidence"] == []


@pytest.mark.parametrize(
    ("repo_relative_path", "expected_fragment"),
    [
        ("../outside.py", "outside the repository"),
        (".git/config", ".git internals"),
        (".env", "environment files"),
        ("src/orders/secret_token.py", "secret-like files"),
        ("assets/logo.png", "binary assets"),
    ],
)
def test_rejects_unsafe_source_scopes_before_reading(
    tmp_path: Path,
    repo_relative_path: str,
    expected_fragment: str,
) -> None:
    result = _run_reading(
        tmp_path,
        question="Summarize this file.",
        source_scope=_scope("file", repo_relative_path),
    )

    _assert_public_shape(result)
    assert result["status"] == "rejected"
    assert expected_fragment in result["answer_text"]
    assert result["evidence"] == []


def test_evidence_collection_returns_repo_relative_safe_rows(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    assignment = {
        "assignment_id": "payment-reader",
        "role": "symbol reader",
        "scope": {
            "kind": "symbol",
            "values": ["PaymentGateway"],
        },
        "questions": ["What does PaymentGateway do?"],
        "required_slots": ["definition"],
    }

    bundle = collect_assignment_evidence(
        repo_root=Path(repository["local_root"]),
        source_scope=_scope(),
        assignment=assignment,
        max_files=6,
        max_excerpt_chars=12000,
    )

    assert bundle.rows
    for row in bundle.rows:
        path = Path(row["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert row["path"] != ".env"
        assert not row["path"].endswith(".png")
        assert len(row["excerpt"]) <= 12000


def test_repository_map_excludes_secret_and_binary_files(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.repository_map import (
        build_repository_map_summary,
    )

    repository = _make_repository(tmp_path)
    summary = build_repository_map_summary(repository, _scope())

    assert "src/orders/service.py" in summary["files"]
    assert ".env" not in summary["files"]
    assert "assets/logo.png" not in summary["files"]


def test_answer_cap_is_enforced_when_public_run_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import supervisor

    def fake_supervisor(_request: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "succeeded",
            "answer_text": "x" * 500,
            "evidence": [],
            "limitations": [],
            "trace_summary": ["reading:test double"],
        }
        return result

    monkeypatch.setattr(supervisor, "run_reading_supervisor", fake_supervisor)
    result = _run_reading(
        tmp_path,
        question="Summarize OrderService.",
    )

    _assert_public_shape(result)
    assert len(result["answer_text"]) <= 1600
