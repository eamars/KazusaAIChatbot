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
    ".list_collection_names(",
    "db[",
)

_RAW_MONGO_METHODS = frozenset({
    "aggregate",
    "bulk_write",
    "count_documents",
    "create_index",
    "delete_many",
    "delete_one",
    "distinct",
    "drop",
    "find",
    "find_one",
    "insert_many",
    "insert_one",
    "list_collection_names",
    "list_indexes",
    "replace_one",
    "update_many",
    "update_one",
})

_COLLECTION_RECEIVER_TERMS = frozenset({
    "coll",
    "collection",
    "database",
    "db",
    "mongo",
})


_REPOSITORY_MODULES = frozenset({
    _SOURCE_ROOT / "calendar_scheduler" / "repository.py",
})


def _production_python_files_outside_db() -> list[Path]:
    files = []
    for path in _SOURCE_ROOT.rglob("*.py"):
        if _DB_ROOT in path.parents:
            continue
        if path.resolve() in _REPOSITORY_MODULES:
            continue
        files.append(path)
    return files


def _relative(path: Path) -> str:
    value = str(path.relative_to(_ROOT))
    return value


def _attribute_call_receiver_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id.lower()]
    if isinstance(node, ast.Attribute):
        names = _attribute_call_receiver_names(node.value)
        names.append(node.attr.lower())
        return names
    return []


def _looks_like_database_find_call(node: ast.Attribute) -> bool:
    receiver_names = _attribute_call_receiver_names(node.value)
    if not receiver_names:
        return True
    for term in _COLLECTION_RECEIVER_TERMS:
        if any(term in receiver_name for receiver_name in receiver_names):
            return True
    return False


def test_db_facade_does_not_export_raw_database_handle() -> None:
    """The public db facade must not expose get_db."""

    assert "get_db" not in db_facade.__all__
    assert not hasattr(db_facade, "get_db")


def test_raw_mongo_find_detector_catches_nested_db_collection_receiver() -> None:
    """Static detector must catch db.collection.find() chains."""

    tree = ast.parse("db.user_memory_units.find({})")
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))

    assert isinstance(call.func, ast.Attribute)
    assert _looks_like_database_find_call(call.func)


def test_raw_mongo_find_detector_allows_string_find_calls() -> None:
    """Static detector must not reject ordinary string ``find`` calls."""

    tree = ast.parse("message_text.find('#napcat')")
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))

    assert isinstance(call.func, ast.Attribute)
    assert not _looks_like_database_find_call(call.func)


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
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            method_name = node.func.attr
            if method_name not in _RAW_MONGO_METHODS:
                continue
            if method_name == "find" and not _looks_like_database_find_call(node.func):
                continue
            violations.append(f"{_relative(path)} calls .{method_name}(")

    assert violations == []
