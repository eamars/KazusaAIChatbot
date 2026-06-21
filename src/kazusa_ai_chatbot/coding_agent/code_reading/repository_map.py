"""Repository-map summaries for bounded code-reading planning."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
    SOURCE_CLASS_ORDER,
    list_scoped_safe_files,
    source_class_for_path,
    summarize_safe_source_file,
)

MAX_REPOSITORY_MAP_FILES = 120
MAX_REPOSITORY_MAP_DIRECTORIES = 40
MAX_SOURCE_CLASS_ITEMS = 36
MAX_TOP_SYMBOLS = 80


def build_repository_map_summary(
    repository: CodeRepositoryRef,
    source_scope: CodeSourceScope,
) -> dict[str, object]:
    """Build a compact safe file map visible to the reading PM.

    Args:
        repository: Successful Phase 0 repository source contract.
        source_scope: Successful Phase 0 source scope.

    Returns:
        Bounded repo-relative file and directory metadata safe to expose in
        planning prompts.
    """

    repo_root = Path(repository["local_root"])
    safe_files = list_scoped_safe_files(
        repo_root=repo_root,
        source_scope=source_scope,
    )
    visible_files = sorted(safe_files)
    directories = _top_directories(visible_files)
    source_classes = _source_classes(
        repo_root=repo_root,
        visible_files=visible_files,
    )
    summary: dict[str, object] = {
        "source_scope_kind": source_scope["kind"],
        "source_scope_path": source_scope["repo_relative_path"],
        "total_safe_files": len(visible_files),
        "files": visible_files[:MAX_REPOSITORY_MAP_FILES],
        "top_directories": directories[:MAX_REPOSITORY_MAP_DIRECTORIES],
        "source_classes": source_classes,
        "top_symbols": _top_symbols(source_classes),
    }
    return summary


def _top_directories(paths: list[str]) -> list[str]:
    directories: list[str] = []
    for path in paths:
        parts = path.split("/")
        if len(parts) <= 1:
            continue
        for index in range(1, len(parts)):
            directory = "/".join(parts[:index])
            if directory not in directories:
                directories.append(directory)
    return directories


def _source_classes(
    *,
    repo_root: Path,
    visible_files: list[str],
) -> dict[str, list[dict[str, object]]]:
    classes: dict[str, list[dict[str, object]]] = {
        source_class: []
        for source_class in SOURCE_CLASS_ORDER
    }
    ranked_files = sorted(
        visible_files,
        key=lambda path: (
            _source_class_rank(source_class_for_path(path)),
            len(path.split("/")),
            path.casefold(),
        ),
    )
    for path in ranked_files:
        source_class = source_class_for_path(path)
        items = classes.setdefault(source_class, [])
        if len(items) >= MAX_SOURCE_CLASS_ITEMS:
            continue
        summary = summarize_safe_source_file(repo_root, path)
        items.append(summary)
    return classes


def _top_symbols(
    source_classes: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    symbols: list[dict[str, object]] = []
    for source_class in SOURCE_CLASS_ORDER:
        for item in source_classes.get(source_class, []):
            defined_symbols = item.get("defined_symbols")
            if not isinstance(defined_symbols, list):
                continue
            for symbol in defined_symbols:
                if not isinstance(symbol, str):
                    continue
                symbols.append({
                    "symbol": symbol,
                    "path": item["path"],
                    "source_class": source_class,
                })
                if len(symbols) >= MAX_TOP_SYMBOLS:
                    return symbols
    return symbols


def _source_class_rank(source_class: str) -> int:
    try:
        return SOURCE_CLASS_ORDER.index(source_class)
    except ValueError:
        return len(SOURCE_CLASS_ORDER)
