"""Static tests for the public database boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import kazusa_ai_chatbot.db as db_facade


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _ROOT / "src" / "kazusa_ai_chatbot"
_DB_ROOT = _SOURCE_ROOT / "db"

_RAW_BACKEND_TOKENS = (
    "db.client",
    ".admin.command",
    ".insert_one(",
    ".insert_many(",
    ".update_one(",
    ".update_many(",
    ".replace_one(",
    ".find(",
    ".find_one(",
    ".delete_one(",
    ".delete_many(",
    ".aggregate(",
    ".count_documents(",
    ".distinct(",
    ".bulk_write(",
    ".create_index(",
    ".list_indexes(",
    ".list_collection_names(",
    ".drop(",
    "db[",
)


def _production_python_files_outside_db() -> list[Path]:
    files = []
    for path in _SOURCE_ROOT.rglob("*.py"):
        if _DB_ROOT in path.parents:
            continue
        files.append(path)
    return files


def _relative(path: Path) -> str:
    value = str(path.relative_to(_ROOT))
    return value


def test_db_facade_does_not_export_raw_database_handle() -> None:
    """The public db facade must not expose get_db."""

    assert "get_db" not in db_facade.__all__
    assert not hasattr(db_facade, "get_db")


def test_production_code_outside_db_does_not_import_private_db_boundary() -> None:
    """Production modules outside db must not import private DB internals."""

    violations: list[str] = []
    for path in _production_python_files_outside_db():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in {"motor", "pymongo"}:
                    violations.append(f"{_relative(path)} imports {module}")
                if module.startswith(("motor.", "pymongo.")):
                    violations.append(f"{_relative(path)} imports {module}")
                if module == "kazusa_ai_chatbot.db._client":
                    violations.append(f"{_relative(path)} imports {module}")
                if module.startswith("kazusa_ai_chatbot.db."):
                    for alias in node.names:
                        if alias.name.startswith("_"):
                            violations.append(
                                f"{_relative(path)} imports private {module}.{alias.name}"
                            )
                if module == "kazusa_ai_chatbot.db":
                    for alias in node.names:
                        if alias.name == "get_db":
                            violations.append(f"{_relative(path)} imports get_db")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"motor", "pymongo"}:
                        violations.append(f"{_relative(path)} imports {alias.name}")
                    if alias.name == "kazusa_ai_chatbot.db._client":
                        violations.append(f"{_relative(path)} imports db._client")

    assert violations == []


def test_production_code_outside_db_has_no_raw_mongo_operations() -> None:
    """Raw Mongo operations belong inside kazusa_ai_chatbot.db only."""

    violations: list[str] = []
    for path in _production_python_files_outside_db():
        text = path.read_text(encoding="utf-8")
        for token in _RAW_BACKEND_TOKENS:
            if token in text:
                violations.append(f"{_relative(path)} contains {token}")

    assert violations == []
