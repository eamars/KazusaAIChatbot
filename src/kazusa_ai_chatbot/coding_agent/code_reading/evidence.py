"""Bounded filesystem evidence collection for code reading."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope
from kazusa_ai_chatbot.coding_agent.code_reading.models import CodeEvidenceRow
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    ReadingPlan,
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

MAX_EVIDENCE_ROWS = 24
MAX_ROWS_PER_TERM = 32
MAX_EXCERPT_CHARS = 700
MAX_FILE_BYTES = 128_000


class EvidenceCollectionError(ValueError):
    """Raised when scoped evidence cannot be safely collected."""


@dataclass(frozen=True)
class EvidenceBundle:
    """Collected evidence plus trace notes and limitations."""

    rows: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]


def collect_evidence(
    *,
    repo_root: Path,
    source_scope: CodeSourceScope,
    plan: ReadingPlan,
) -> EvidenceBundle:
    """Collect bounded evidence for a reading plan."""

    root = repo_root.resolve(strict=True)
    scoped_files = _scoped_files(root, source_scope)
    trace_summary = [
        f"reading:scope={source_scope['kind']}",
        "reading:listed files with rg --files",
    ]

    if plan.broad:
        return EvidenceBundle(
            rows=[],
            limitations=[],
            trace_summary=[*trace_summary, "reading:broad question needs scope"],
        )

    if plan.summary_scope or not plan.terms:
        rows = _summarize_files(root, scoped_files, plan)
        limitations = _limitations_for_rows(rows, plan)
        return EvidenceBundle(
            rows=rows,
            limitations=limitations,
            trace_summary=trace_summary,
        )

    rows = _search_terms(
        root=root,
        scoped_files=scoped_files,
        terms=plan.terms,
        plan=plan,
    )
    if plan.needs_tests:
        rows = [
            row for row in rows if row["path"].startswith("tests/")
        ] or rows

    rows = _rank_rows(rows, plan)[:MAX_EVIDENCE_ROWS]
    limitations = _limitations_for_rows(rows, plan)
    trace_summary.append("reading:searched files with rg -n --json")
    return EvidenceBundle(
        rows=rows,
        limitations=limitations,
        trace_summary=trace_summary,
    )


def find_definition_paths(
    *,
    repo_root: Path,
    source_scope: CodeSourceScope,
    symbol: str,
) -> list[str]:
    """Find repository-relative paths defining a class or function symbol."""

    root = repo_root.resolve(strict=True)
    scoped_files = _scoped_files(root, source_scope)
    definition_symbols = [symbol]
    if "." in symbol:
        definition_symbols.extend(part for part in symbol.split(".") if part)
    definition_prefixes = tuple(
        prefix
        for item in definition_symbols
        for prefix in (
            f"class {item}",
            f"def {item}",
            f"async def {item}",
        )
    )
    paths: list[str] = []
    for relative_path in scoped_files:
        text = _read_text_file(root / relative_path)
        if text is None:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(definition_prefixes):
                paths.append(_to_posix(relative_path))
                break
    return sorted(set(paths))


def _scoped_files(root: Path, source_scope: CodeSourceScope) -> list[Path]:
    all_files = _rg_files(root)
    scoped_path = source_scope.get("repo_relative_path")
    if scoped_path is None:
        candidates = all_files
    else:
        scope_root = ensure_path_inside(root / scoped_path, root)
        if source_scope["kind"] == "file":
            relative_path = scope_root.relative_to(root)
            candidates = [relative_path]
        else:
            candidates = [
                item for item in all_files
                if (root / item).is_relative_to(scope_root)
            ]

    safe_files = [
        item for item in candidates
        if _is_safe_relative_file(item)
    ]
    return safe_files


def _rg_files(root: Path) -> list[Path]:
    try:
        completed = subprocess.run(
            [
                "rg",
                "--files",
                "--hidden",
                "-g",
                "!.git/*",
                "-g",
                "!.tmp_pytest/**",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return _walk_files(root)

    if completed.returncode not in (0, 1):
        return _walk_files(root)

    files: list[Path] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        files.append(Path(line))
    return files


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path.relative_to(root))
    return files


def _search_terms(
    *,
    root: Path,
    scoped_files: list[Path],
    terms: tuple[str, ...],
    plan: ReadingPlan,
) -> list[CodeEvidenceRow]:
    scoped_set = {_to_posix(path) for path in scoped_files}
    rows: list[CodeEvidenceRow] = []
    seen: set[tuple[str, int, str]] = set()
    for term in terms:
        term_rows = 0
        for row in _rg_search(root, term):
            if row["path"] not in scoped_set:
                continue
            key = (row["path"], row["line_start"], term.casefold())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            term_rows += 1
            if term_rows >= MAX_ROWS_PER_TERM:
                break

    if rows:
        return rows

    return _summarize_files(root, scoped_files, plan)


def _rg_search(root: Path, term: str) -> list[CodeEvidenceRow]:
    try:
        completed = subprocess.run(
            [
                "rg",
                "-n",
                "--json",
                "--hidden",
                "-g",
                "!.git/*",
                "-g",
                "!.tmp_pytest/**",
                "-F",
                "--",
                term,
                ".",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return _python_search(root, term)

    if completed.returncode not in (0, 1):
        return _python_search(root, term)

    rows: list[CodeEvidenceRow] = []
    for line in completed.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "match":
            continue
        data = payload.get("data", {})
        path_text = data.get("path", {}).get("text")
        line_number = data.get("line_number")
        if not isinstance(path_text, str) or not isinstance(line_number, int):
            continue
        relative_path = Path(path_text)
        if not _is_safe_relative_file(relative_path):
            continue
        row = _row_for_match(root, relative_path, line_number, term)
        if row is not None:
            rows.append(row)
    return rows


def _python_search(root: Path, term: str) -> list[CodeEvidenceRow]:
    rows: list[CodeEvidenceRow] = []
    for relative_path in _rg_files(root):
        if not _is_safe_relative_file(relative_path):
            continue
        text = _read_text_file(root / relative_path)
        if text is None:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            if term in line:
                row = _row_for_match(root, relative_path, index, term)
                if row is not None:
                    rows.append(row)
    return rows


def _summarize_files(
    root: Path,
    scoped_files: list[Path],
    plan: ReadingPlan,
) -> list[CodeEvidenceRow]:
    rows: list[CodeEvidenceRow] = []
    for relative_path in scoped_files:
        row = _row_for_file_summary(root, relative_path, plan)
        if row is not None:
            rows.append(row)
        if len(rows) >= MAX_EVIDENCE_ROWS:
            break
    return rows


def _row_for_match(
    root: Path,
    relative_path: Path,
    line_number: int,
    term: str,
) -> CodeEvidenceRow | None:
    text = _read_text_file(root / relative_path)
    if text is None:
        return None

    lines = text.splitlines()
    if line_number < 1 or line_number > len(lines):
        return None

    start = max(1, line_number - 3)
    end = min(len(lines), line_number + 3)
    excerpt = "\n".join(lines[start - 1:end])
    row: CodeEvidenceRow = {
        "path": _to_posix(relative_path),
        "line_start": start,
        "line_end": end,
        "symbol_or_topic": term,
        "excerpt": _cap_excerpt(excerpt),
        "reason": f"Matched query term: {term}",
    }
    return row


def _row_for_file_summary(
    root: Path,
    relative_path: Path,
    plan: ReadingPlan,
) -> CodeEvidenceRow | None:
    text = _read_text_file(root / relative_path)
    if text is None:
        return None

    selected_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith(("class ", "def ", "async def "))
            or stripped.startswith(("#", "from ", "import ", "@"))
            or "=" in stripped
        ):
            selected_lines.append(line)
        if len(selected_lines) >= 8:
            break

    if not selected_lines:
        selected_lines = text.splitlines()[:8]

    excerpt = "\n".join(selected_lines)
    if not excerpt.strip():
        return None

    row: CodeEvidenceRow = {
        "path": _to_posix(relative_path),
        "line_start": 1,
        "line_end": min(8, max(1, len(text.splitlines()))),
        "symbol_or_topic": plan.family,
        "excerpt": _cap_excerpt(excerpt),
        "reason": "Selected bounded file summary evidence.",
    }
    return row


def _rank_rows(
    rows: list[CodeEvidenceRow],
    plan: ReadingPlan,
) -> list[CodeEvidenceRow]:
    family = plan.family

    def score(row: CodeEvidenceRow) -> tuple[int, str, int]:
        path = row["path"]
        value = 100
        if family == "test_coverage_mapping" and path.startswith("tests/"):
            value -= 50
        if family == "docs_to_code_consistency" and path == "README.md":
            value -= 30
        if family == "build_run_reading" and (
            path == "README.md" or "deploy" in path
        ):
            value -= 30
        if family == "architecture_responsibility" and (
            "background_work" in path or "architecture" in path
        ):
            value -= 30
        if family == "dependency_usage" and (
            "integrations" in path or "routes" in path
        ):
            value -= 30
        if family == "feature_pipeline_explanation" and (
            "adapter" in path
            or "service" in path
            or "brain_service" in path
            or "descriptor" in path
            or "pipeline" in path
            or "episode" in path
            or "history" in path
            or "attachments" in path
            or "message_envelope" in path
            or "persona_supervisor2_msg_decontexualizer" in path
            or "persona_supervisor2_cognition" in path
            or "routes" in path
        ):
            value -= 30
        if family == "feature_pipeline_explanation" and path.startswith("src/"):
            value -= 40
        if family == "feature_pipeline_explanation":
            value += _feature_path_penalty(path)
            value -= _feature_topic_bonus(row["symbol_or_topic"])
            value -= _feature_exact_path_bonus(path)
        if family == "feature_pipeline_explanation" and path.startswith("tests/"):
            value += 25
        return (value, path, row["line_start"])

    sorted_rows = sorted(rows, key=score)
    if family == "feature_pipeline_explanation":
        return _rank_feature_rows(sorted_rows)

    deduped: list[CodeEvidenceRow] = []
    seen_lines: set[tuple[str, int]] = set()
    for row in sorted_rows:
        key = (row["path"], row["line_start"])
        if key in seen_lines:
            continue
        seen_lines.add(key)
        deduped.append(row)
    return deduped


def _rank_feature_rows(
    sorted_rows: list[CodeEvidenceRow],
) -> list[CodeEvidenceRow]:
    stage_topics = [
        "base64_data",
        "user_multimedia_input",
        "multimedia_descriptor_agent",
        "VISION_DESCRIPTOR_LLM",
        "image_observation",
        "update_conversation_attachment_descriptions",
        "<image>",
    ]
    selected: list[CodeEvidenceRow] = []
    seen_lines: set[tuple[str, int]] = set()
    path_counts: dict[str, int] = {}

    for topic in stage_topics:
        topic_rows = [
            row for row in sorted_rows
            if row["symbol_or_topic"] == topic
        ]
        for row in sorted(
            topic_rows,
            key=lambda item, stage_topic=topic: _stage_topic_score(
                item,
                stage_topic,
            ),
        ):
            if _feature_row_blocked(row, seen_lines, path_counts):
                continue
            _select_feature_row(row, selected, seen_lines, path_counts)
            break

    for row in sorted_rows:
        if _feature_row_blocked(row, seen_lines, path_counts):
            continue
        _select_feature_row(row, selected, seen_lines, path_counts)

    return selected


def _feature_row_blocked(
    row: CodeEvidenceRow,
    seen_lines: set[tuple[str, int]],
    path_counts: dict[str, int],
) -> bool:
    key = (row["path"], row["line_start"])
    if key in seen_lines:
        return True
    return path_counts.get(row["path"], 0) >= 3


def _select_feature_row(
    row: CodeEvidenceRow,
    selected: list[CodeEvidenceRow],
    seen_lines: set[tuple[str, int]],
    path_counts: dict[str, int],
) -> None:
    selected.append(row)
    seen_lines.add((row["path"], row["line_start"]))
    path_counts[row["path"]] = path_counts.get(row["path"], 0) + 1


def _stage_topic_score(row: CodeEvidenceRow, topic: str) -> tuple[int, str, int]:
    path = row["path"]
    score = _feature_path_penalty(path)
    if topic == "base64_data":
        score += _path_preference(
            path,
            [
                "src/adapters/",
                "src/kazusa_ai_chatbot/message_envelope/",
                "src/kazusa_ai_chatbot/service.py",
                "src/kazusa_ai_chatbot/brain_service/contracts.py",
            ],
        )
    elif topic == "user_multimedia_input":
        score += _path_preference(
            path,
            [
                "src/kazusa_ai_chatbot/service.py",
                "src/kazusa_ai_chatbot/brain_service/graph.py",
                "src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py",
            ],
        )
    elif topic == "multimedia_descriptor_agent":
        score += _path_preference(
            path,
            [
                "src/kazusa_ai_chatbot/brain_service/graph.py",
                "src/kazusa_ai_chatbot/service.py",
                "src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py",
            ],
        )
    elif topic == "VISION_DESCRIPTOR_LLM":
        score += _path_preference(
            path,
            [
                "src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py",
                "src/kazusa_ai_chatbot/config.py",
                "src/kazusa_ai_chatbot/llm_interface/route_report.py",
            ],
        )
    elif topic == "image_observation":
        score += _path_preference(
            path,
            [
                "src/kazusa_ai_chatbot/cognition_episode.py",
                "src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py",
                "src/kazusa_ai_chatbot/cognition_chain_core/prompt_selection.py",
                "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py",
            ],
        )
    elif topic == "update_conversation_attachment_descriptions":
        score += _path_preference(
            path,
            [
                "src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py",
                "src/kazusa_ai_chatbot/db/conversation.py",
            ],
        )
    elif topic == "<image>":
        score += _path_preference(
            path,
            [
                "src/kazusa_ai_chatbot/utils.py",
                "src/kazusa_ai_chatbot/message_envelope/prompt_projection.py",
                "src/kazusa_ai_chatbot/rag/prompt_projection.py",
                "src/adapters/napcat_qq_adapter/cq_projection.py",
            ],
        )
    return (score, path, row["line_start"])


def _path_preference(path: str, prefixes_or_paths: list[str]) -> int:
    for index, prefix_or_path in enumerate(prefixes_or_paths):
        if path == prefix_or_path or path.startswith(prefix_or_path):
            return -100 + (index * 10)
    return 0


def _feature_path_penalty(path: str) -> int:
    if "/coding_agent/" in path:
        return 90
    if path.startswith("development_plans/"):
        return 60
    if path.startswith("tests/"):
        return 40
    if path.startswith(("README", "docs/", "AGENTS.md")):
        return 30
    return 0


def _feature_topic_bonus(topic: str) -> int:
    priorities = {
        "base64_data": 50,
        "user_multimedia_input": 55,
        "multimedia_descriptor_agent": 60,
        "VISION_DESCRIPTOR_LLM": 55,
        "image_observation": 45,
        "update_conversation_attachment_descriptions": 45,
        "<image>": 45,
    }
    return priorities.get(topic, 0)


def _feature_exact_path_bonus(path: str) -> int:
    priority_paths = {
        "src/kazusa_ai_chatbot/brain_service/graph.py": 70,
        "src/kazusa_ai_chatbot/service.py": 70,
        (
            "src/kazusa_ai_chatbot/nodes/"
            "persona_supervisor2_msg_decontexualizer.py"
        ): 80,
        "src/kazusa_ai_chatbot/cognition_episode.py": 65,
        "src/kazusa_ai_chatbot/message_envelope/attachment_handlers/image.py": 60,
        "src/kazusa_ai_chatbot/message_envelope/prompt_projection.py": 55,
        "src/kazusa_ai_chatbot/utils.py": 55,
        "src/kazusa_ai_chatbot/db/conversation.py": 50,
        "src/adapters/discord_adapter.py": 55,
        "src/adapters/napcat_qq_adapter/attachments.py": 55,
    }
    return priority_paths.get(path, 0)


def _limitations_for_rows(
    rows: list[CodeEvidenceRow],
    plan: ReadingPlan,
) -> list[str]:
    limitations: list[str] = []
    if not rows:
        limitations.append("No bounded source evidence matched the question.")
        return limitations

    if len(rows) >= MAX_EVIDENCE_ROWS:
        limitations.append("Evidence was capped to the top bounded matches.")

    if plan.family == "lifecycle_cache_persistence":
        limitations.append(
            "Only checked-in cache and persistence evidence was available."
        )
    elif plan.family == "general_reading":
        limitations.append(
            "Answer is limited to repository evidence matched by query terms."
        )
    return limitations


def _is_safe_relative_file(relative_path: Path) -> bool:
    parts = relative_path.parts
    if not parts or relative_path.is_absolute() or ".." in parts:
        return False
    if ".git" in parts:
        return False

    path_text = _to_posix(relative_path)
    name = parts[-1].casefold()
    if name == ".env" or name.startswith(".env."):
        return False
    if is_secret_like_path(path_text):
        return False
    if is_binary_like_path(path_text):
        return False
    return True


def _read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None

    if b"\x00" in data:
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _cap_excerpt(excerpt: str) -> str:
    if len(excerpt) <= MAX_EXCERPT_CHARS:
        return excerpt
    return excerpt[:MAX_EXCERPT_CHARS].rstrip()


def _to_posix(path: Path) -> str:
    return path.as_posix()
