"""Repository-map summaries for bounded code-reading planning."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
    list_scoped_safe_files,
)

MAX_REPOSITORY_MAP_FILES = 120
MAX_REPOSITORY_MAP_DIRECTORIES = 40


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
    summary: dict[str, object] = {
        "source_scope_kind": source_scope["kind"],
        "source_scope_path": source_scope["repo_relative_path"],
        "total_safe_files": len(visible_files),
        "files": visible_files[:MAX_REPOSITORY_MAP_FILES],
        "top_directories": directories[:MAX_REPOSITORY_MAP_DIRECTORIES],
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
