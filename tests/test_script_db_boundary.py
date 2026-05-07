"""Static tests for script database access boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_ROOT = _ROOT / "src" / "scripts"

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

_RAW_QUERY_TOKENS = (
    '"$exists"',
    '"$or"',
    '"$pull"',
    '"$set"',
    '"$unset"',
    "'$exists'",
    "'$or'",
    "'$pull'",
    "'$set'",
    "'$unset'",
)


def _script_python_files() -> list[Path]:
    files = list(_SCRIPT_ROOT.rglob("*.py"))
    return files


def _relative(path: Path) -> str:
    value = str(path.relative_to(_ROOT))
    return value


def test_scripts_do_not_import_raw_or_private_db_boundary() -> None:
    """Scripts must use public DB helpers instead of get_db or db._client."""

    violations: list[str] = []
    for path in _script_python_files():
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


def test_scripts_do_not_construct_raw_mongo_query_or_update_dsl() -> None:
    """Scripts should pass semantic parameters to DB helpers."""

    violations: list[str] = []
    for path in _script_python_files():
        text = path.read_text(encoding="utf-8")
        for token in _RAW_QUERY_TOKENS:
            if token in text:
                violations.append(f"{_relative(path)} contains {token}")

    assert violations == []


def test_scripts_do_not_use_raw_mongo_operations() -> None:
    """Scripts should call script-facing public DB helpers for DB work."""

    violations: list[str] = []
    for path in _script_python_files():
        text = path.read_text(encoding="utf-8")
        for token in _RAW_BACKEND_TOKENS:
            if token in text:
                violations.append(f"{_relative(path)} contains {token}")

    assert violations == []
