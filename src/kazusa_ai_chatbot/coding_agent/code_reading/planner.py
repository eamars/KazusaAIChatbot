"""Deterministic safety checks for the code-reading subagent."""

import re
from pathlib import PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope


def rejection_reason(
    question: str,
    *,
    read_only_context_handoff: bool = False,
) -> str | None:
    """Return a read-only rejection reason, if the request is unsupported.

    Args:
        question: Prompt-facing reading request.
        read_only_context_handoff: Whether write or execution terms are
            requirements context from the top-level coding supervisor, not a
            direct instruction for the reading workflow to mutate or execute.
    """

    normalized = question.casefold()
    if not read_only_context_handoff:
        write_patterns = [
            "apply a patch",
            "write a patch",
            "propose patches",
            "rewrite ",
            "modify ",
            "edit ",
            "change the code",
        ]
        if any(pattern in normalized for pattern in write_patterns):
            return "Code reading is read-only and cannot write patches."

        execution_patterns = [
            "run pytest",
            "run tests",
            "execute ",
            "shell command",
            "install ",
            "pip install",
            "npm install",
        ]
        if any(pattern in normalized for pattern in execution_patterns):
            return "Code reading cannot execute commands or install packages."

    secret_patterns = [
        r"(?<![a-z])\.env(?:\b|$)",
        r"\bread\b.*\bsecret\b",
        r"\binspect\b.*\bsecret\b",
        r"\bdump\b.*\btoken\b",
        r"\bread\b.*\bcredential\b",
        r"\bprivate[_ ]key\b",
    ]
    if any(re.search(pattern, normalized) for pattern in secret_patterns):
        return "Code reading cannot inspect secrets or environment files."

    raw_dump_patterns = [
        "dump the full raw",
        "full raw contents",
        "entire raw file",
    ]
    if any(pattern in normalized for pattern in raw_dump_patterns):
        return "Code reading returns bounded evidence, not raw file dumps."

    binary_patterns = [
        "binary pixels",
        "analyze the binary",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    ]
    if any(pattern in normalized for pattern in binary_patterns):
        return "Code reading cannot analyze binary assets."

    if "git@github.com" in normalized or "private repo" in normalized:
        return "Code reading supports only public or local source contracts."

    certification_patterns = [
        "certify",
        "legally compliant",
        "secure",
    ]
    if any(pattern in normalized for pattern in certification_patterns):
        return "Code reading cannot certify legal or security status."

    external_current_patterns = [
        "latest ",
        "today",
        "current cve",
        "cve status",
    ]
    if any(pattern in normalized for pattern in external_current_patterns):
        return "Code reading cannot answer current external facts requiring web evidence."

    return None


def source_scope_rejection_reason(source_scope: CodeSourceScope) -> str | None:
    """Validate source-scope shape before filesystem reads."""

    repo_relative_path = source_scope.get("repo_relative_path")
    if repo_relative_path is None:
        return None

    path = PurePosixPath(repo_relative_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return "Resolved source scope is outside the repository."
    if any(part == ".git" for part in path.parts):
        return "Code reading cannot inspect .git internals."

    lowered_name = path.name.casefold()
    if lowered_name == ".env" or lowered_name.startswith(".env."):
        return "Code reading cannot inspect environment files."
    if is_secret_like_path(str(path)):
        return "Code reading cannot inspect secret-like files."
    if is_binary_like_path(str(path)):
        return "Code reading cannot analyze binary assets."

    return None


def is_secret_like_path(path: str) -> bool:
    """Return whether a repository-relative path is likely secret material."""

    lowered = path.casefold()
    secret_fragments = [
        "secret",
        "credential",
        "private_key",
        "id_rsa",
        "token",
    ]
    secret_suffixes = [
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    ]
    has_fragment = any(item in lowered for item in secret_fragments)
    has_suffix = any(lowered.endswith(item) for item in secret_suffixes)
    return has_fragment or has_suffix


def is_binary_like_path(path: str) -> bool:
    """Return whether a path suffix normally denotes binary content."""

    lowered = path.casefold()
    binary_suffixes = [
        ".7z",
        ".avif",
        ".bmp",
        ".class",
        ".dll",
        ".exe",
        ".gif",
        ".ico",
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
        ".pyc",
        ".so",
        ".webp",
        ".zip",
    ]
    binary_like = any(lowered.endswith(item) for item in binary_suffixes)
    return binary_like
