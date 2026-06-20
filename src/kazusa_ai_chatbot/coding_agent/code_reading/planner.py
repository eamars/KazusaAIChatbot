"""Question planning for the code-reading subagent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope

_CODE_TOKEN_RE = re.compile(r"`([^`]+)`|[A-Za-z_][A-Za-z0-9_./:-]*")

_STOP_WORDS = {
    "a",
    "after",
    "all",
    "and",
    "are",
    "as",
    "before",
    "by",
    "code",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "project",
    "repository",
    "source",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}

_IMAGE_TERMS = [
    "base64_data",
    "user_multimedia_input",
    "multimedia_descriptor_agent",
    "VISION_DESCRIPTOR_LLM",
    "image_observation",
    "update_conversation_attachment_descriptions",
    "<image>",
    "ImagePipeline",
    "attachments",
    "image",
    "images",
]


@dataclass(frozen=True)
class ReadingPlan:
    """Concrete read-only plan for bounded evidence collection."""

    family: str
    terms: tuple[str, ...]
    summary_scope: bool
    broad: bool
    compare_symbols: tuple[str, ...]
    symbol: str | None
    needs_tests: bool


def rejection_reason(question: str) -> str | None:
    """Return a read-only rejection reason, if the request is unsupported."""

    normalized = question.casefold()
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
        return "Phase 1 code reading is read-only and cannot write patches."

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
        return "Phase 1 code reading cannot execute commands or install packages."

    secret_patterns = [
        ".env",
        "secret",
        "token",
        "credential",
        "private key",
    ]
    if any(pattern in normalized for pattern in secret_patterns):
        return "Phase 1 code reading cannot inspect secrets or environment files."

    raw_dump_patterns = [
        "dump the full raw",
        "full raw contents",
        "entire raw file",
    ]
    if any(pattern in normalized for pattern in raw_dump_patterns):
        return "Phase 1 code reading returns bounded evidence, not raw file dumps."

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
        return "Phase 1 code reading cannot analyze binary assets."

    if "git@github.com" in normalized or "private repo" in normalized:
        return "Phase 1 supports only Phase 0 public or local source contracts."

    certification_patterns = [
        "certify",
        "legally compliant",
        "secure",
    ]
    if any(pattern in normalized for pattern in certification_patterns):
        return "Phase 1 cannot certify legal or security status."

    external_current_patterns = [
        "latest ",
        "today",
        "current cve",
        "cve status",
    ]
    if any(pattern in normalized for pattern in external_current_patterns):
        return "Phase 1 cannot answer current external facts requiring web evidence."

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
        return "Phase 1 code reading cannot inspect .git internals."

    lowered_name = path.name.casefold()
    if lowered_name == ".env" or lowered_name.startswith(".env."):
        return "Phase 1 code reading cannot inspect environment files."
    if is_secret_like_path(str(path)):
        return "Phase 1 code reading cannot inspect secret-like files."
    if is_binary_like_path(str(path)):
        return "Phase 1 code reading cannot analyze binary assets."

    return None


def build_plan(question: str, source_scope: CodeSourceScope) -> ReadingPlan:
    """Build a deterministic evidence plan from the question and source scope."""

    normalized = question.casefold()
    terms = _terms_from_question(question)
    family = "general_reading"
    summary_scope = False
    broad = False
    needs_tests = False

    if "everything" in normalized and "repository" in normalized:
        broad = True
        family = "broad_repository_read"
    elif "test" in normalized and "cover" in normalized:
        family = "test_coverage_mapping"
        needs_tests = True
        terms.extend(["test_", "image"])
    elif "compare" in normalized:
        family = "intra_repo_comparison"
    elif "readme" in normalized and (
        "match" in normalized or "consistent" in normalized
    ):
        family = "docs_to_code_consistency"
        terms.extend(["README", "/chat", "ChatRequest"])
    elif "run this project" in normalized or "how do i run" in normalized:
        family = "build_run_reading"
        terms.extend(["fixture-service", "CLI_COMMAND", "Run with"])
    elif "depend on" in normalized or "depends on" in normalized:
        family = "static_impact_read"
    elif any(
        item in normalized
        for item in ("mongo", "openai", "fastapi", "external integration")
    ):
        family = "dependency_usage"
        terms.extend(["MongoClient", "OpenAI", "FastAPI"])
    elif any(item in normalized for item in ("failure", "cached", "stored")):
        family = "lifecycle_cache_persistence"
        terms.extend(["cache_failure", "MongoClient", "persist", "stored"])
    elif "created and consumed" in normalized or "state" in normalized:
        family = "state_model_reading"
    elif "where is" in normalized or "used" in normalized:
        family = "definition_usage_search"
    elif "what does" in normalized:
        family = "symbol_explanation"
    elif "request shape" in normalized or "endpoint" in normalized:
        family = "api_contract_lookup"
        terms.extend(["/chat", "ChatRequest", "attachments"])
    elif "owns" in normalized or "responsibility" in normalized:
        family = "architecture_responsibility"
        terms.extend(["background_work", "route_background_work"])
    elif (
        source_scope["kind"] in ("file", "directory")
        and "summarize" in normalized
    ):
        family = "scope_summary"
        summary_scope = True
    elif any(item in normalized for item in ("image", "images", "读图", "图片")):
        family = "feature_pipeline_explanation"
        terms.extend(_IMAGE_TERMS)

    compare_symbols = _comparison_symbols(question)
    symbol = _primary_symbol(question)
    if symbol is not None:
        terms.extend(_symbol_terms(symbol, question))
    terms = _dedupe_terms(terms)
    return ReadingPlan(
        family=family,
        terms=tuple(terms),
        summary_scope=summary_scope,
        broad=broad,
        compare_symbols=tuple(compare_symbols),
        symbol=symbol,
        needs_tests=needs_tests,
    )


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
    return any(item in lowered for item in secret_fragments) or any(
        lowered.endswith(item)
        for item in secret_suffixes
    )


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
    return any(lowered.endswith(item) for item in binary_suffixes)


def _terms_from_question(question: str) -> list[str]:
    terms: list[str] = []
    for match in _CODE_TOKEN_RE.finditer(question):
        token = match.group(1) or match.group(0)
        token = token.strip(".,?!;:()[]{}")
        if not token:
            continue
        lowered = token.casefold()
        if lowered in _STOP_WORDS:
            continue
        if token.startswith("http"):
            continue
        terms.append(token)

    if any(item in question for item in ("读图", "图片", "图像")):
        terms.extend(_IMAGE_TERMS)

    return terms


def _primary_symbol(question: str) -> str | None:
    tokens = _terms_from_question(question)
    for token in tokens:
        if "." in token and not token.startswith("/"):
            return token
        if token and (token[0].isupper() or "_" in token):
            return token
    return None


def _symbol_terms(symbol: str, question: str) -> list[str]:
    terms = [symbol]
    if "." in symbol:
        terms.extend(part for part in symbol.split(".") if part)
    normalized = question.casefold()
    if "what does" in normalized:
        for token in terms[:]:
            if token and token[0].isupper():
                terms.append(f"class {token}")
            else:
                terms.append(f"def {token}")
    return terms


def _comparison_symbols(question: str) -> list[str]:
    if "compare" not in question.casefold():
        return []
    symbols = []
    for token in _terms_from_question(question):
        if token and token[0].isupper():
            symbols.append(token)
    return _dedupe_terms(symbols)


def _dedupe_terms(terms: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped
