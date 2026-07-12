"""Deterministic proposal-bound verification plan derivation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionSpec,
)
from kazusa_ai_chatbot.coding_agent.code_patching.models import PatchArtifact
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    _parse_patch_artifacts,
)
from kazusa_ai_chatbot.coding_agent.path_classification import is_test_path
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PM_LLM_API_KEY,
    CODING_AGENT_PM_LLM_BASE_URL,
    CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PM_LLM_MODEL,
    CODING_AGENT_PM_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

EXECUTION_PLAN_SCHEMA_VERSION = "coding_execution_plan.v1"
DEFAULT_COMPILE_TIMEOUT_SECONDS = 30
DEFAULT_PYTEST_TIMEOUT_SECONDS = 60
MAX_ADDITIVE_SAFE_TEST_PATHS = 32
ADDITIVE_EXECUTION_SPEC_PROMPT = '''
Select optional additive verification only from the supplied safe test paths.
Return an empty list unless the user explicitly requests an extra check.

# Output Format
Return strict JSON with one field:
- `pytest_selectors`: a list of zero or more exact relative paths from
  `safe_test_paths`. Do not return any other paths, selectors, commands, or
  fields.
'''
_additive_execution_spec_llm = LLInterface()
_additive_execution_spec_config = LLMCallConfig(
    stage_name=__name__, route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL, api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL, temperature=0.1, top_p=0.7, top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None, timeout_seconds=120,
    thinking=LLMThinkingConfig(enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED),
)


async def extract_additive_execution_specs(
    *,
    user_request: str,
    candidate_root: Path,
) -> list[CodeExecutionSpec]:
    """Use the coding specialist to select explicit additive test checks."""

    safe_tests = [
        path.relative_to(candidate_root).as_posix()
        for path in candidate_root.rglob("test_*.py")
        if "tests" in path.relative_to(candidate_root).parts
    ][:MAX_ADDITIVE_SAFE_TEST_PATHS]
    if not user_request.strip() or not safe_tests:
        return []
    payload = {"user_request": user_request, "safe_test_paths": safe_tests}
    messages = [
        SystemMessage(content=ADDITIVE_EXECUTION_SPEC_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    response = await _additive_execution_spec_llm.ainvoke(
        messages, config=_additive_execution_spec_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping):
        return []
    selectors = parsed.get("pytest_selectors")
    spec: CodeExecutionSpec = {"tool": "pytest", "pytest_selectors": selectors}
    if not _is_safe_extra_spec(spec, candidate_root=candidate_root):
        return []
    return [spec]


def patch_artifact_digest(patch_artifacts: list[PatchArtifact]) -> str:
    """Return a stable digest for ordered reviewed patch artifacts."""

    canonical_artifacts: list[dict[str, object]] = []
    for artifact in patch_artifacts:
        canonical_artifacts.append({
            "artifact_id": artifact.get("artifact_id", ""),
            "base": artifact.get("base", ""),
            "diff_text": artifact.get("diff_text", ""),
            "files": artifact.get("files", []),
            "summary": artifact.get("summary", ""),
        })
    encoded = json.dumps(
        canonical_artifacts,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest


def derive_base_execution_plan(
    *,
    candidate_root: Path,
    patch_artifacts: list[PatchArtifact],
    run_id: str,
    source_identity: Mapping[str, object],
    proposal_revision: int,
) -> dict[str, object]:
    """Derive safe compile and test checks from the proposal's changed files."""

    parsed = _parse_patch_artifacts(
        patch_artifacts=patch_artifacts,
        max_files=64,
        max_diff_chars=64_000,
    )
    changed_paths = [
        path
        for path in parsed["files"]
        if isinstance(path, str)
    ]
    python_paths = [
        path
        for path in changed_paths
        if path.endswith(".py")
    ]
    test_paths = _safe_test_paths(
        candidate_root=candidate_root,
        changed_paths=changed_paths,
    )
    base_specs: list[CodeExecutionSpec] = []
    if python_paths:
        base_specs.append({
            "tool": "python_compileall",
            "paths": python_paths,
            "timeout_seconds": DEFAULT_COMPILE_TIMEOUT_SECONDS,
        })
    if test_paths:
        base_specs.append({
            "tool": "pytest",
            "pytest_selectors": test_paths,
            "timeout_seconds": DEFAULT_PYTEST_TIMEOUT_SECONDS,
        })
    limitations: list[str] = []
    if python_paths and not test_paths:
        limitations.append("no_focused_test_discovered")
    plan = {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "plan_id": _plan_id(
            run_id=run_id,
            source_identity=source_identity,
            proposal_revision=proposal_revision,
            artifact_digest=patch_artifact_digest(patch_artifacts),
        ),
        "origin": "changed_files",
        "run_id": run_id,
        "source_identity": dict(source_identity),
        "proposal_revision": proposal_revision,
        "patch_artifact_digest": patch_artifact_digest(patch_artifacts),
        "changed_paths": changed_paths,
        "base_specs": _without_default_timeouts(base_specs),
        "limitations": limitations,
    }
    return plan


def validate_execution_plan_binding(
    *,
    plan: Mapping[str, object],
    run_id: str,
    source_identity: Mapping[str, object],
    proposal_revision: int,
    patch_artifact_digest: str,
) -> str:
    """Return a public error when an execution plan is stale or incomplete."""

    if plan.get("schema_version") != EXECUTION_PLAN_SCHEMA_VERSION:
        return "Execution plan schema is unsupported."
    if plan.get("run_id") != run_id:
        return "Execution plan run binding is stale."
    if plan.get("source_identity") != dict(source_identity):
        return "Execution plan source identity is stale."
    if plan.get("proposal_revision") != proposal_revision:
        return "Execution plan proposal revision is stale."
    if plan.get("patch_artifact_digest") != patch_artifact_digest:
        return "Execution plan patch artifact digest is stale."
    return ""


def merge_additive_execution_specs(
    *,
    plan: Mapping[str, object],
    extra_specs: list[CodeExecutionSpec],
    candidate_root: Path,
) -> list[CodeExecutionSpec]:
    """Merge validated user-requested checks without replacing the base plan."""

    base_value = plan.get("base_specs")
    base_specs = _execution_specs(base_value)
    merged_specs = list(base_specs)
    for extra_spec in extra_specs:
        if not _is_safe_extra_spec(extra_spec, candidate_root=candidate_root):
            continue
        if extra_spec in merged_specs:
            continue
        merged_specs.append(extra_spec)
    return merged_specs


def _safe_test_paths(*, candidate_root: Path, changed_paths: list[str]) -> list[str]:
    safe_paths: list[str] = []
    for changed_path in changed_paths:
        if is_test_path(changed_path):
            _append_changed_test_path(safe_paths, changed_path)
            continue
        companion_path = _conventional_test_companion(changed_path)
        if companion_path:
            _append_if_existing(safe_paths, candidate_root, companion_path)
    return safe_paths


def _conventional_test_companion(path: str) -> str:
    path_value = PurePosixPath(path)
    if path_value.suffix != ".py":
        return ""
    if path_value.parts and path_value.parts[0] == "src":
        candidate = PurePosixPath("tests") / f"test_{path_value.stem}.py"
        return candidate.as_posix()
    candidate = PurePosixPath("tests") / f"test_{path_value.stem}.py"
    return candidate.as_posix()


def _append_if_existing(paths: list[str], candidate_root: Path, path: str) -> None:
    if path in paths or not _candidate_file_exists(candidate_root, path):
        return
    paths.append(path)


def _append_changed_test_path(paths: list[str], path: str) -> None:
    """Keep a proposal-created test path even before candidate materialization."""

    if path in paths:
        return
    paths.append(path)


def _candidate_file_exists(candidate_root: Path, path: str) -> bool:
    candidate_path = candidate_root / PurePosixPath(path)
    return candidate_path.is_file()


def _plan_id(
    *,
    run_id: str,
    source_identity: Mapping[str, object],
    proposal_revision: int,
    artifact_digest: str,
) -> str:
    binding = {
        "run_id": run_id,
        "source_identity": dict(source_identity),
        "proposal_revision": proposal_revision,
        "patch_artifact_digest": artifact_digest,
    }
    encoded = json.dumps(binding, ensure_ascii=False, sort_keys=True).encode("utf-8")
    plan_id = hashlib.sha256(encoded).hexdigest()
    return plan_id


def _without_default_timeouts(
    specs: list[CodeExecutionSpec],
) -> list[CodeExecutionSpec]:
    normalized_specs: list[CodeExecutionSpec] = []
    for spec in specs:
        normalized_spec = dict(spec)
        normalized_spec.pop("timeout_seconds", None)
        normalized_specs.append(normalized_spec)
    return normalized_specs


def _execution_specs(value: object) -> list[CodeExecutionSpec]:
    if not isinstance(value, list):
        return []
    specs: list[CodeExecutionSpec] = []
    for row in value:
        if not isinstance(row, Mapping):
            continue
        specs.append(dict(row))
    return specs


def _is_safe_extra_spec(
    spec: Mapping[str, object],
    *,
    candidate_root: Path,
) -> bool:
    if spec.get("tool") == "python_compileall":
        paths = spec.get("paths")
        return _safe_existing_paths(paths, candidate_root=candidate_root)
    if spec.get("tool") == "pytest":
        selectors = spec.get("pytest_selectors")
        return _safe_existing_paths(selectors, candidate_root=candidate_root)
    return False


def _safe_existing_paths(value: object, *, candidate_root: Path) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for path_value in value:
        if not isinstance(path_value, str) or not path_value.strip():
            return False
        path = path_value.split("::", 1)[0].replace("\\", "/")
        path_value_object = PurePosixPath(path)
        if path_value_object.is_absolute() or ".." in path_value_object.parts:
            return False
        if not _candidate_file_exists(candidate_root, path):
            return False
    return True
