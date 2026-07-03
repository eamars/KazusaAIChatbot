from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_reading.planner import rejection_reason


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


def test_read_only_handoff_treats_write_terms_as_context() -> None:
    assert rejection_reason("Apply a patch to rewrite OrderService.") is not None

    reason = rejection_reason(
        "Read current evidence. User request context: apply a patch and run tests.",
        read_only_context_handoff=True,
    )

    assert reason is None
    assert (
        rejection_reason(
            "Read current evidence. User request context: inspect .env.",
            read_only_context_handoff=True,
        )
        is not None
    )


def test_reading_pm_payload_compacts_large_repository_map() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
        MAX_PM_ITEM_EXCERPT_CHARS,
        MAX_PM_REPO_MAP_DIRECTORIES,
        MAX_PM_REPO_MAP_FILES,
        MAX_PM_SOURCE_CLASS_ITEMS,
        MAX_PM_TOP_SYMBOLS,
        MAX_PM_QUESTION_CHARS,
        _pm_payload,
    )

    long_excerpt = "class Example:\n    pass\n" * 80
    source_items = [
        {
            "path": f"src/module_{index}.py",
            "source_class": "python",
            "defined_symbols": [f"Symbol{nested}" for nested in range(20)],
            "imported_modules": [f"module_{nested}" for nested in range(20)],
            "summary_excerpt": long_excerpt,
        }
        for index in range(80)
    ]
    pm_input = {
        "question": "Read current evidence.\n" + ("requirements\n" * 5000),
        "repository_summary": {"repo": "fixture"},
        "source_scope": {"kind": "repository", "repo_relative_path": None},
        "repo_map_summary": {
            "source_scope_kind": "repository",
            "source_scope_path": None,
            "total_safe_files": 200,
            "files": [f"src/module_{index}.py" for index in range(200)],
            "top_directories": [f"src/package_{index}" for index in range(80)],
            "source_classes": {"python": source_items},
            "top_symbols": [
                {
                    "symbol": f"Symbol{index}",
                    "path": f"src/module_{index}.py",
                    "source_class": "python",
                }
                for index in range(120)
            ],
        },
        "previous_reports": [],
    }

    payload = _pm_payload(pm_input)
    repo_map = payload["repo_map_summary"]

    assert len(payload["question"]) <= MAX_PM_QUESTION_CHARS
    assert len(repo_map["files"]) == MAX_PM_REPO_MAP_FILES
    assert len(repo_map["top_directories"]) == MAX_PM_REPO_MAP_DIRECTORIES
    assert len(repo_map["source_classes"]["python"]) == MAX_PM_SOURCE_CLASS_ITEMS
    assert (
        len(repo_map["source_classes"]["python"][0]["summary_excerpt"])
        <= MAX_PM_ITEM_EXCERPT_CHARS
    )
    assert len(repo_map["top_symbols"]) == MAX_PM_TOP_SYMBOLS


def test_reading_synthesis_payload_compacts_large_evidence() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.synthesizer import (
        MAX_SYNTHESIS_EVIDENCE_EXCERPT_CHARS,
        MAX_SYNTHESIS_EVIDENCE_ROWS,
        MAX_SYNTHESIS_FACTS,
        MAX_SYNTHESIS_FILES_READ,
        MAX_SYNTHESIS_NEXT_HOPS,
        MAX_SYNTHESIS_QUESTION_CHARS,
        MAX_SYNTHESIS_REPORTS,
        _synthesis_payload,
    )

    long_excerpt = "def example():\n    return 'value'\n" * 120
    evidence = [
        {
            "path": f"src/module_{index}.py",
            "line_start": 1,
            "line_end": 20,
            "symbol_or_topic": f"Symbol{index}",
            "excerpt": long_excerpt,
            "reason": "source evidence",
        }
        for index in range(80)
    ]
    reports = [
        {
            "assignment_id": f"assignment-{index}",
            "status": "succeeded",
            "files_read": [f"src/module_{nested}.py" for nested in range(20)],
            "facts": [
                {
                    "kind": "behavior",
                    "summary": "fact summary " * 200,
                    "evidence_refs": [f"src/module_{nested}.py:1-2" for nested in range(20)],
                }
                for _ in range(20)
            ],
            "evidence": evidence,
            "open_questions": ["open question" for _ in range(20)],
            "discovered_symbols": [f"Symbol{nested}" for nested in range(20)],
            "candidate_next_hops": [
                {
                    "reason": "next hop" * 100,
                    "scope": {
                        "kind": "file",
                        "values": [f"src/module_{nested}.py"],
                    },
                }
                for nested in range(20)
            ],
        }
        for index in range(12)
    ]

    payload = _synthesis_payload(
        question="requirements\n" * 5000,
        pm_decision={
            "status": "sufficient",
            "intent": "architecture_overview",
            "required_slots": [],
            "assignments": [],
            "missing_slots": [],
        },
        programmer_reports=reports,
        evidence=evidence,
        limitations=[],
        repository_summary={"repo": "fixture"},
        preferred_language="English",
        max_answer_chars=1600,
    )

    assert len(payload["question"]) <= MAX_SYNTHESIS_QUESTION_CHARS
    assert len(payload["programmer_reports"]) == MAX_SYNTHESIS_REPORTS
    assert (
        len(payload["programmer_reports"][0]["files_read"])
        == MAX_SYNTHESIS_FILES_READ
    )
    assert len(payload["programmer_reports"][0]["facts"]) == MAX_SYNTHESIS_FACTS
    assert (
        len(payload["programmer_reports"][0]["candidate_next_hops"])
        == MAX_SYNTHESIS_NEXT_HOPS
    )
    assert len(payload["selected_evidence"]) == MAX_SYNTHESIS_EVIDENCE_ROWS
    assert (
        len(payload["selected_evidence"][0]["excerpt"])
        <= MAX_SYNTHESIS_EVIDENCE_EXCERPT_CHARS
    )


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
    repo_root = Path(repository["local_root"])
    (repo_root / ".agents" / "skills").mkdir(parents=True)
    (repo_root / ".agents" / "skills" / "tool.py").write_text(
        "def support_tool() -> None:\n    pass\n",
        encoding="utf-8",
    )
    summary = build_repository_map_summary(repository, _scope())

    assert "src/orders/service.py" in summary["files"]
    assert ".env" not in summary["files"]
    assert "assets/logo.png" not in summary["files"]
    assert ".agents/skills/tool.py" not in summary["files"]


def test_repository_map_reads_source_under_ignored_parent(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.repository_map import (
        build_repository_map_summary,
    )

    (tmp_path / ".gitignore").write_text(
        "ignored_source/\n",
        encoding="utf-8",
    )
    repo_root = tmp_path / "ignored_source"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "assets").mkdir()
    (repo_root / "src" / "runtime.py").write_text(
        "def run() -> str:\n    return 'ok'\n",
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
        "repo": "ignored-source",
        "source_url": "local://fixture/ignored-source",
        "requested_ref": None,
        "resolved_ref": "generated",
        "current_commit": "generated-sha256:fixture",
        "default_branch": "generated",
        "local_root": str(repo_root),
        "storage_kind": "managed_download",
        "managed_checkout": True,
        "workspace_root": str(tmp_path),
        "cache_key": None,
        "dirty_state": "clean",
    }

    summary = build_repository_map_summary(repository, _scope())

    assert "src/runtime.py" in summary["files"]
    assert ".env" not in summary["files"]
    assert "assets/logo.png" not in summary["files"]


def test_repository_intelligence_classifies_sources_and_symbols(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.repository_map import (
        build_repository_map_summary,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    (repo_root / "tests").mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "tests" / "test_service.py").write_text(
        "def test_submit_order_flow():\n    assert True\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "orders.md").write_text(
        "The order service is documented here.\n",
        encoding="utf-8",
    )
    (repo_root / "scripts" / "inspect_orders.py").write_text(
        "print('inspect orders')\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "orders" / "tools").mkdir()
    (repo_root / "src" / "orders" / "tools" / "runtime.py").write_text(
        "def run_order_tooling() -> None:\n    pass\n",
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n",
        encoding="utf-8",
    )

    summary = build_repository_map_summary(repository, _scope())

    source_classes = summary["source_classes"]
    implementation_paths = [
        item["path"]
        for item in source_classes["implementation"]
    ]
    test_paths = [item["path"] for item in source_classes["tests"]]
    docs_paths = [item["path"] for item in source_classes["docs"]]
    scripts_paths = [item["path"] for item in source_classes["scripts"]]
    config_paths = [item["path"] for item in source_classes["config"]]

    assert "src/orders/service.py" in implementation_paths
    assert "src/orders/tools/runtime.py" in implementation_paths
    assert "tests/test_service.py" in test_paths
    assert "docs/orders.md" in docs_paths
    assert "scripts/inspect_orders.py" in scripts_paths
    assert "pyproject.toml" in config_paths

    service_summary = next(
        item
        for item in source_classes["implementation"]
        if item["path"] == "src/orders/service.py"
    )
    assert "OrderService" in service_summary["defined_symbols"]
    assert "orders.gateway" in service_summary["imported_modules"]


def test_search_evidence_prefers_implementation_over_tests_and_docs(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    (repo_root / "tests").mkdir()
    (repo_root / "docs" / "handoff.md").write_text(
        "handoff appears in operational documentation.\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_handoff.py").write_text(
        "def test_handoff_contract():\n    assert 'handoff'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "orders" / "handoff.py").write_text(
        "def commit_handoff(event: dict) -> str:\n    return event['status']\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "handoff-search",
        "role": "flow reader",
        "scope": {
            "kind": "search",
            "values": ["handoff"],
        },
        "questions": ["How does the handoff flow work?"],
        "required_slots": ["implementation flow"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert bundle.files_read[0] == "src/orders/handoff.py"


def test_search_evidence_uses_assignment_questions_to_rank_broad_terms(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    for index in range(6):
        (flow_root / f"a_noise_{index}.py").write_text(
            "\n".join(
                [
                    f"def generic_notice_{index}() -> str:",
                    "    return 'notice was recorded'",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    (flow_root / "z_decision.py").write_text(
        "\n".join(
            [
                "def decide_notice_channels(account: dict) -> list[str]:",
                "    if account['suspended'] or account['offline']:",
                "        return ['email', 'webhook']",
                "    return ['inbox']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "notice-routing",
        "role": "flow reader",
        "scope": {
            "kind": "search",
            "values": ["notice"],
        },
        "questions": [
            "When an account may be suspended or offline, how are "
            "email, webhook, and inbox notices decided?"
        ],
        "required_slots": ["channel decision logic"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert bundle.files_read[0] == "src/orders/flow/z_decision.py"


def test_search_evidence_keeps_distinct_regions_from_same_file(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    early_lines = [
        "def decide_notice_channels(account: dict) -> list[str]:",
        "    # notice suspended offline email webhook inbox decision",
        "    if account['suspended'] or account['offline']:",
        "        return ['email', 'webhook']",
    ]
    spacer_lines = [f"    audit_value_{index} = {index}" for index in range(50)]
    later_lines = [
        "def archive_notice_delivery(account: dict) -> bool:",
        "    archive_marker = account['archive_marker']",
        "    return bool(archive_marker)",
    ]
    (flow_root / "decision.py").write_text(
        "\n".join([*early_lines, *spacer_lines, *later_lines]) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "notice-routing",
        "role": "flow reader",
        "scope": {
            "kind": "search",
            "values": ["notice"],
        },
        "questions": [
            "How are suspended, offline, email, webhook, inbox, and "
            "archive notice decisions handled?"
        ],
        "required_slots": ["channel decision logic"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert any("archive_marker" in row["excerpt"] for row in bundle.rows)


def test_search_evidence_keeps_more_than_three_source_regions(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    source_lines: list[str] = []
    for marker in ["alpha", "beta", "gamma", "delta"]:
        source_lines.extend([
            f"def notice_decision_checkpoint_{marker}(account: dict) -> str:",
            "    # notice decision checkpoint",
            f"    {marker}_outcome = account['{marker}']",
            f"    return {marker}_outcome",
        ])
        source_lines.extend(
            f"padding_{marker}_{index} = {index}" for index in range(60)
        )
    (flow_root / "multi_region.py").write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "notice-multi-region",
        "role": "flow reader",
        "scope": {
            "kind": "search",
            "values": ["notice"],
        },
        "questions": [
            "How are notice decision checkpoints handled across the workflow?"
        ],
        "required_slots": ["checkpoint decision logic"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=20000,
    )

    assert any("delta_outcome" in row["excerpt"] for row in bundle.rows)


def test_search_evidence_prefers_state_transition_branches(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    source_lines: list[str] = []
    for marker in ["alpha", "beta", "gamma", "delta", "epsilon"]:
        source_lines.extend([
            f"def workflow_state_context_{marker}(workflow_run):",
            "    # WorkflowRun evaluate item states terminal state context",
            f"    {marker}_state_note = workflow_run.state",
            f"    return {marker}_state_note",
        ])
        source_lines.extend(
            f"state_padding_{marker}_{index} = {index}"
            for index in range(35)
        )
    source_lines.extend([
        "def workflow_state_terminal_decision(workflow_run):",
        "    # WorkflowRun evaluate item states terminal state context",
        "    if any(item.state in ItemState.failed_states for item in workflow_run.items):",
        "        workflow_run.set_state(WorkflowState.FAILED)",
        "    elif all(item.state in ItemState.success_states for item in workflow_run.items):",
        "        workflow_run.set_state(WorkflowState.SUCCESS)",
        "    return workflow_run.state",
    ])
    (flow_root / "state_machine.py").write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "state-transition",
        "role": "flow reader",
        "scope": {
            "kind": "search",
            "values": ["WorkflowRun", "state"],
        },
        "questions": [
            "How does WorkflowRun evaluate item states to determine terminal state?"
        ],
        "required_slots": ["terminal state decision logic"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=20000,
    )

    assert any("WorkflowState.FAILED" in row["excerpt"] for row in bundle.rows)
    assert any("WorkflowState.SUCCESS" in row["excerpt"] for row in bundle.rows)


def test_search_evidence_splits_multiword_scope_values(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    (flow_root / "routing.py").write_text(
        "\n".join([
            "def get_route_handler(call):",
            "    response = run_endpoint_function(call)",
            "    return response",
        ]) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "route-handler",
        "role": "reader",
        "scope": {
            "kind": "search",
            "values": ["OrderRoute get_route_handler exception"],
        },
        "questions": ["Which source handles it?"],
        "required_slots": ["result"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert bundle.files_read[0] == "src/orders/flow/routing.py"
    assert any("run_endpoint_function" in row["excerpt"] for row in bundle.rows)


def test_file_scope_reads_bounded_rows_from_large_source_file(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    padding = "x" * 180_000
    (flow_root / "large_routing.py").write_text(
        "\n".join([
            "def get_route_handler(call):",
            "    response = run_endpoint_function(call)",
            "    return response",
            f"PADDING = '{padding}'",
        ]) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "large-route-handler",
        "role": "reader",
        "scope": {
            "kind": "file",
            "values": ["src/orders/flow/large_routing.py"],
        },
        "questions": ["How does get_route_handler call the endpoint?"],
        "required_slots": ["endpoint call logic"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert bundle.files_read == ["src/orders/flow/large_routing.py"]
    assert any("run_endpoint_function" in row["excerpt"] for row in bundle.rows)
    assert all(len(row["excerpt"]) <= 12000 for row in bundle.rows)


def test_search_evidence_keeps_nearby_class_execution_method(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    source_lines = [
        "class LoadRecord:",
        "    @classmethod",
        "    def INPUT_TYPES(cls):",
        "        return {'required': {'record': ('STRING',)}}",
    ]
    source_lines.extend(f"    config_{index} = {index}" for index in range(12))
    source_lines.extend([
        "    FUNCTION = 'load_record'",
        "",
        "    def load_record(self, record):",
        "        payload = open_record(record)",
        "        return payload",
    ])
    (flow_root / "loader.py").write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "load-record-node",
        "role": "reader",
        "scope": {
            "kind": "search",
            "values": ["LoadRecord"],
        },
        "questions": ["Which method implements the LoadRecord node?"],
        "required_slots": ["execution method"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert any("def load_record" in row["excerpt"] for row in bundle.rows)
    assert any("open_record" in row["excerpt"] for row in bundle.rows)


def test_search_evidence_keeps_nearby_branch_inside_excerpt_window(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
        collect_assignment_evidence,
    )

    repository = _make_repository(tmp_path)
    repo_root = Path(repository["local_root"])
    flow_root = repo_root / "src" / "orders" / "flow"
    flow_root.mkdir()
    source_lines = [
        "def decide_notice_channels(account: dict) -> list[str]:",
        "    '''Decide notice behavior from a hierarchy of settings.",
        "    The notice setting can override broad defaults.",
        "    The branch below applies account-specific rules.",
        "    '''",
        "    base_setting = account['base_setting']",
        "    stream_setting = account['stream_setting']",
        "    effective_setting = stream_setting or base_setting",
        "    if effective_setting == 'silent':",
        "        return []",
        "    if account['mode'] == 'suppress':",
        "        final_action = 'suppress'",
        "        return [final_action]",
        "    return ['email']",
    ]
    (flow_root / "window.py").write_text(
        "\n".join(source_lines) + "\n",
        encoding="utf-8",
    )
    assignment = {
        "assignment_id": "notice-window",
        "role": "flow reader",
        "scope": {
            "kind": "search",
            "values": ["notice"],
        },
        "questions": [
            "How does the notice hierarchy handle archive policy branches?"
        ],
        "required_slots": ["branch decision logic"],
    }

    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=_scope(),
        assignment=assignment,
        max_files=3,
        max_excerpt_chars=12000,
    )

    assert any("final_action" in row["excerpt"] for row in bundle.rows)


def test_answer_cap_is_enforced_when_public_run_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import agent

    def fake_supervisor(_request: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "succeeded",
            "answer_text": "x" * 500,
            "evidence": [],
            "limitations": [],
            "trace_summary": ["reading:test double"],
        }
        return result

    monkeypatch.setattr(agent, "run_reading_supervisor", fake_supervisor)
    result = _run_reading(
        tmp_path,
        question="Summarize OrderService.",
    )

    _assert_public_shape(result)
    assert len(result["answer_text"]) <= 1600
