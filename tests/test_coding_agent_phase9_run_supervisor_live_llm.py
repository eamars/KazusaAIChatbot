"""Live LLM gates for durable coding-run supervision."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, TypedDict

import pytest

from kazusa_ai_chatbot.coding_agent.code_patching.models import (
    PatchArtifact,
    PatchOperation,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    compile_patch_operations,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_run_supervisor")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_run_supervisor_workspace")
MAX_ANSWER_CHARS = 6000
MAX_ARTIFACT_CHARS = 70000
MAX_DIFF_CHARS = 70000
MAX_FILES = 12


class CodingRunGate(TypedDict, total=False):
    """One live coding-run gate contract."""

    gate_id: str
    title: str
    objective_type: str
    instruction: str
    source_files: dict[str, str]
    execution_specs: list[dict[str, object]]
    initial_operations: list[PatchOperation]
    expected_status_after_start: str
    expected_terminal_status: str
    expected_changed_paths: list[str]
    protected_source_paths: list[str]
    behavior_rubric: list[str]
    forbidden_failure_modes: list[str]


GATE_01: CodingRunGate = {
    "gate_id": "run_gate_01_read_only_persistence",
    "title": "Read-only run ledger persistence",
    "objective_type": "read_only",
    "instruction": (
        "Explain how the score loader handles blank lines and malformed rows. "
        "Cite the source file and tests that prove the behavior."
    ),
    "source_files": {
        "scores.py": (
            "def load_scores(text):\n"
            "    scores = []\n"
            "    for line_number, line in enumerate(text.splitlines(), start=1):\n"
            "        stripped = line.strip()\n"
            "        if not stripped:\n"
            "            continue\n"
            "        name, value = stripped.split(',', 1)\n"
            "        scores.append((name.strip(), int(value)))\n"
            "    return scores\n"
        ),
        "tests/test_scores.py": (
            "import pytest\n\n"
            "from scores import load_scores\n\n\n"
            "def test_load_scores_skips_blank_lines():\n"
            "    assert load_scores('Ada,10\\n\\nGrace,8') == [\n"
            "        ('Ada', 10),\n"
            "        ('Grace', 8),\n"
            "    ]\n\n\n"
            "def test_load_scores_rejects_malformed_rows():\n"
            "    with pytest.raises(ValueError):\n"
            "        load_scores('Ada')\n"
        ),
    },
    "expected_status_after_start": "completed",
    "expected_terminal_status": "completed",
    "expected_changed_paths": [],
    "protected_source_paths": ["scores.py", "tests/test_scores.py"],
    "behavior_rubric": [
        "The run creates a durable ledger and records source/evidence events.",
        "The answer is grounded in repository evidence, not generic CSV advice.",
        "A fresh get call returns the same public run status after persistence.",
    ],
    "forbidden_failure_modes": [
        "The public response exposes absolute source or workspace roots.",
        "The run fabricates behavior not present in scores.py or tests.",
        "The get call loses events or evidence from the started run.",
    ],
}

GATE_02: CodingRunGate = {
    "gate_id": "run_gate_02_proposal_awaits_approval",
    "title": "Patch proposal pauses before side effects",
    "objective_type": "propose_patch",
    "instruction": (
        "Modify the existing name utility so normalize_name strips leading "
        "and trailing whitespace, collapses internal whitespace to one space, "
        "and keeps the original letter casing. Update focused tests."
    ),
    "source_files": {
        "name_tools.py": (
            "def normalize_name(value):\n"
            "    return value.strip()\n"
        ),
        "tests/test_name_tools.py": (
            "from name_tools import normalize_name\n\n\n"
            "def test_normalize_name_strips_outer_whitespace():\n"
            "    assert normalize_name('  Ada Lovelace  ') == 'Ada Lovelace'\n"
        ),
    },
    "expected_status_after_start": "awaiting_approval",
    "expected_terminal_status": "awaiting_approval",
    "expected_changed_paths": ["name_tools.py"],
    "protected_source_paths": ["tests/test_name_tools.py"],
    "behavior_rubric": [
        "The live PM/programmer flow produces a source-backed patch proposal.",
        "The run pauses at approval without applying patches or running tests.",
        "The ledger records proposal-ready and awaiting-approval events.",
    ],
    "forbidden_failure_modes": [
        "The run applies the patch during start without structured approval.",
        "The proposal changes only tests while runtime behavior is unchanged.",
        "The public ledger stores raw diffs, local roots, or command output.",
    ],
}

GATE_03: CodingRunGate = {
    "gate_id": "run_gate_03_approve_verify_success",
    "title": "Approval continuation verifies a patch",
    "objective_type": "propose_patch",
    "instruction": (
        "Repair slugify so punctuation is removed, whitespace collapses to "
        "single dashes, output is lowercase, and empty results raise "
        "ValueError. Do not modify the provided verification tests."
    ),
    "source_files": {
        "slug_tools.py": (
            "def slugify(value):\n"
            "    return value.strip().lower().replace(' ', '-')\n"
        ),
        "tests/test_slug_tools.py": (
            "import pytest\n\n"
            "from slug_tools import slugify\n\n\n"
            "def test_slugify_removes_punctuation_and_collapses_spaces():\n"
            "    assert slugify(' Hello,   World!! ') == 'hello-world'\n\n\n"
            "def test_slugify_rejects_empty_result():\n"
            "    with pytest.raises(ValueError):\n"
            "        slugify('!!!')\n"
        ),
    },
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_slug_tools.py"],
        "timeout_seconds": 20,
    }],
    "expected_status_after_start": "awaiting_approval",
    "expected_terminal_status": "completed",
    "expected_changed_paths": ["slug_tools.py"],
    "protected_source_paths": ["tests/test_slug_tools.py"],
    "behavior_rubric": [
        "Start creates a reviewable proposal and waits for approval.",
        "Continuation applies the approved proposal only into a managed copy.",
        "Focused pytest succeeds, or the verifier repairs within the run cap.",
    ],
    "forbidden_failure_modes": [
        "Continuation mutates the original source checkout.",
        "The run reports completion without a succeeded final execution.",
        "Verification edits the protected tests instead of slug_tools.py.",
    ],
}

GATE_04: CodingRunGate = {
    "gate_id": "run_gate_04_cancel_after_proposal",
    "title": "Cancellation preserves proposal history",
    "objective_type": "propose_patch",
    "instruction": (
        "Modify the receipt formatter so format_receipt accepts an optional "
        "currency keyword argument and defaults to USD. Keep the existing "
        "two-decimal formatting behavior."
    ),
    "source_files": {
        "receipt.py": (
            "def format_receipt(total):\n"
            "    return f'USD {total:.2f}'\n"
        ),
        "tests/test_receipt.py": (
            "from receipt import format_receipt\n\n\n"
            "def test_format_receipt_default_currency():\n"
            "    assert format_receipt(12) == 'USD 12.00'\n"
        ),
    },
    "expected_status_after_start": "awaiting_approval",
    "expected_terminal_status": "cancelled",
    "expected_changed_paths": ["receipt.py"],
    "protected_source_paths": ["tests/test_receipt.py"],
    "behavior_rubric": [
        "The run creates a proposal through the live coding-agent roles.",
        "Cancellation is deterministic and does not invoke apply or execute.",
        "The public get response preserves proposal summary and cancel event.",
    ],
    "forbidden_failure_modes": [
        "Cancellation deletes the run ledger or proposal summary.",
        "A cancelled run can still be continued into apply or execution.",
        "The public response exposes absolute managed workspace paths.",
    ],
}

GATE_05: CodingRunGate = {
    "gate_id": "run_gate_05_seeded_repair_attempt_ledger",
    "title": "Hard mixed seeded repair ledger",
    "objective_type": "verify_repair",
    "instruction": (
        "Repair the release feed utility so page fetching supports a file "
        "cache, --refresh-cache bypasses the cache, --cache-dir selects cache "
        "storage, and --timeout is passed to fetch_page. Do not modify the "
        "provided verification tests."
    ),
    "source_files": {
        "releasefeed/__init__.py": "",
        "releasefeed/fetch.py": (
            "from urllib.request import urlopen\n\n\n"
            "def fetch_page(url):\n"
            "    with urlopen(url) as response:\n"
            "        return response.read().decode('utf-8')\n"
        ),
        "releasefeed/cli.py": (
            "import argparse\n\n"
            "from .fetch import fetch_page\n\n\n"
            "def build_parser():\n"
            "    parser = argparse.ArgumentParser(prog='releasefeed')\n"
            "    parser.add_argument('url')\n"
            "    return parser\n\n\n"
            "def main(argv=None):\n"
            "    args = build_parser().parse_args(argv)\n"
            "    print(fetch_page(args.url))\n"
        ),
        "tests/test_fetch.py": (
            "from releasefeed.fetch import fetch_page\n\n\n"
            "class FakeResponse:\n"
            "    def __enter__(self):\n"
            "        return self\n\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n\n"
            "    def read(self):\n"
            "        return b'fresh'\n\n\n"
            "def test_fetch_page_uses_cache(tmp_path, monkeypatch):\n"
            "    calls = []\n\n"
            "    def fake_urlopen(url, timeout=10):\n"
            "        calls.append((url, timeout))\n"
            "        return FakeResponse()\n\n"
            "    monkeypatch.setattr('releasefeed.fetch.urlopen', fake_urlopen)\n"
            "    cache_dir = tmp_path / 'cache'\n"
            "    assert fetch_page('https://example.test/feed', cache_dir=cache_dir) == 'fresh'\n"
            "    assert fetch_page('https://example.test/feed', cache_dir=cache_dir) == 'fresh'\n"
            "    assert calls == [('https://example.test/feed', 10)]\n\n\n"
            "def test_fetch_page_refresh_cache_bypasses_existing(tmp_path, monkeypatch):\n"
            "    calls = []\n\n"
            "    def fake_urlopen(url, timeout=2):\n"
            "        calls.append((url, timeout))\n"
            "        return FakeResponse()\n\n"
            "    monkeypatch.setattr('releasefeed.fetch.urlopen', fake_urlopen)\n"
            "    cache_dir = tmp_path / 'cache'\n"
            "    fetch_page('https://example.test/feed', cache_dir=cache_dir)\n"
            "    fetch_page(\n"
            "        'https://example.test/feed',\n"
            "        cache_dir=cache_dir,\n"
            "        refresh_cache=True,\n"
            "        timeout=2,\n"
            "    )\n"
            "    assert calls[-1] == ('https://example.test/feed', 2)\n"
        ),
        "tests/test_cli.py": (
            "from releasefeed.cli import main\n\n\n"
            "def test_cli_passes_cache_flags(tmp_path, monkeypatch, capsys):\n"
            "    seen = {}\n\n"
            "    def fake_fetch_page(url, *, cache_dir=None, refresh_cache=False, timeout=10):\n"
            "        seen['url'] = url\n"
            "        seen['cache_dir'] = cache_dir\n"
            "        seen['refresh_cache'] = refresh_cache\n"
            "        seen['timeout'] = timeout\n"
            "        return 'body'\n\n"
            "    monkeypatch.setattr('releasefeed.cli.fetch_page', fake_fetch_page)\n"
            "    main([\n"
            "        '--cache-dir', str(tmp_path),\n"
            "        '--refresh-cache',\n"
            "        '--timeout', '3',\n"
            "        'https://example.test/feed',\n"
            "    ])\n"
            "    assert capsys.readouterr().out.strip() == 'body'\n"
            "    assert seen == {\n"
            "        'url': 'https://example.test/feed',\n"
            "        'cache_dir': tmp_path,\n"
            "        'refresh_cache': True,\n"
            "        'timeout': 3,\n"
            "    }\n"
        ),
        "README.md": (
            "# Release Feed\n\n"
            "Fetch and print a release feed URL.\n"
        ),
    },
    "initial_operations": [
        {
            "operation_id": "seed-fetch-cache-only",
            "kind": "replace_file_small",
            "path": "releasefeed/fetch.py",
            "content": (
                "import hashlib\n"
                "from pathlib import Path\n"
                "from urllib.request import urlopen\n\n\n"
                "def _cache_path(cache_dir, url):\n"
                "    digest = hashlib.sha256(url.encode('utf-8')).hexdigest()\n"
                "    return Path(cache_dir) / f'{digest}.html'\n\n\n"
                "def fetch_page(url, *, cache_dir=None, refresh_cache=False, timeout=10):\n"
                "    if cache_dir is not None:\n"
                "        path = _cache_path(cache_dir, url)\n"
                "        if path.exists() and not refresh_cache:\n"
                "            return path.read_text(encoding='utf-8')\n"
                "    with urlopen(url, timeout=timeout) as response:\n"
                "        body = response.read().decode('utf-8')\n"
                "    if cache_dir is not None:\n"
                "        path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        path.write_text(body, encoding='utf-8')\n"
                "    return body\n"
            ),
            "summary": "Seed patch adds fetch cache support but omits CLI flags.",
        },
    ],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_fetch.py", "tests/test_cli.py"],
        "timeout_seconds": 30,
    }],
    "expected_status_after_start": "completed",
    "expected_terminal_status": "completed",
    "expected_changed_paths": [
        "releasefeed/fetch.py",
        "releasefeed/cli.py",
    ],
    "protected_source_paths": ["tests/test_fetch.py", "tests/test_cli.py"],
    "behavior_rubric": [
        "The run records the seeded proposal, failed execution, repair, and final execution.",
        "Repair targets runtime source paths, not verification tests.",
        "The ledger survives a get call with public-safe attempt summaries.",
    ],
    "forbidden_failure_modes": [
        "The repair edits tests or README instead of source behavior.",
        "The run loses the first failed execution attempt from its public ledger.",
        "The public response exposes raw command output or absolute paths.",
    ],
}


async def test_coding_run_live_gate_01_read_only_state_persistence() -> None:
    """Run a read-only coding session and reload its public state."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run, start_coding_run

    gate_workspace, source_root, source_identity = _prepare_gate_workspace(GATE_01)
    source_hashes_before = _hash_source_files(source_root)

    response = await start_coding_run(
        _start_request(
            gate=GATE_01,
            gate_workspace=gate_workspace,
            source_root=source_root,
        )
    )
    reloaded_response = await get_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": response["run_id"],
    })

    source_hashes_after = _hash_source_files(source_root)
    raw_evidence = _raw_evidence(
        gate=GATE_01,
        source_root=source_root,
        gate_workspace=gate_workspace,
        source_identity=source_identity,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        response=response,
        reloaded_response=reloaded_response,
    )
    _write_json(_raw_evidence_path(GATE_01), raw_evidence)

    _assert_common_run_response(
        gate=GATE_01,
        response=response,
        source_root=source_root,
        gate_workspace=gate_workspace,
    )
    assert response["status"] == "completed"
    assert response["answer_text"]
    assert response["evidence"]
    assert reloaded_response["run_id"] == response["run_id"]
    assert reloaded_response["status"] == response["status"]
    assert raw_evidence["source_tree_unchanged"] is True


async def test_coding_run_live_gate_02_patch_proposal_awaits_approval() -> None:
    """Create a patch proposal and stop before apply or execution."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run, start_coding_run

    gate_workspace, source_root, source_identity = _prepare_gate_workspace(GATE_02)
    source_hashes_before = _hash_source_files(source_root)

    response = await start_coding_run(
        _start_request(
            gate=GATE_02,
            gate_workspace=gate_workspace,
            source_root=source_root,
        )
    )
    reloaded_response = await get_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": response["run_id"],
    })

    source_hashes_after = _hash_source_files(source_root)
    raw_evidence = _raw_evidence(
        gate=GATE_02,
        source_root=source_root,
        gate_workspace=gate_workspace,
        source_identity=source_identity,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        response=response,
        reloaded_response=reloaded_response,
    )
    _write_json(_raw_evidence_path(GATE_02), raw_evidence)

    _assert_common_run_response(
        gate=GATE_02,
        response=response,
        source_root=source_root,
        gate_workspace=gate_workspace,
    )
    assert response["status"] == "awaiting_approval"
    assert response["patch_artifacts"]
    assert not response["apply_attempts"]
    assert not response["execution_attempts"]
    _assert_expected_changed_paths(gate=GATE_02, response=response)
    assert raw_evidence["source_tree_unchanged"] is True


async def test_coding_run_live_gate_03_approve_and_verify_success() -> None:
    """Continue an approved run through managed apply and verification."""

    from kazusa_ai_chatbot.coding_agent import (
        continue_coding_run,
        get_coding_run,
        start_coding_run,
    )

    gate_workspace, source_root, source_identity = _prepare_gate_workspace(GATE_03)
    source_hashes_before = _hash_source_files(source_root)

    start_response = await start_coding_run(
        _start_request(
            gate=GATE_03,
            gate_workspace=gate_workspace,
            source_root=source_root,
        )
    )
    response = await continue_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": start_response["run_id"],
        "action": "approve_and_verify",
        "approval": _approval(GATE_03["gate_id"]),
        "execution_specs": GATE_03["execution_specs"],
        "repair_attempt_limit": 1,
    })
    reloaded_response = await get_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": response["run_id"],
    })

    source_hashes_after = _hash_source_files(source_root)
    raw_evidence = _raw_evidence(
        gate=GATE_03,
        source_root=source_root,
        gate_workspace=gate_workspace,
        source_identity=source_identity,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        response=response,
        reloaded_response=reloaded_response,
        start_response=start_response,
    )
    _write_json(_raw_evidence_path(GATE_03), raw_evidence)

    assert start_response["status"] == "awaiting_approval"
    _assert_common_run_response(
        gate=GATE_03,
        response=response,
        source_root=source_root,
        gate_workspace=gate_workspace,
    )
    assert response["status"] == "completed"
    _assert_expected_changed_paths(gate=GATE_03, response=response)
    _assert_final_execution_succeeded(response)
    _assert_protected_paths_not_changed(gate=GATE_03, response=response)
    assert raw_evidence["source_tree_unchanged"] is True


async def test_coding_run_live_gate_04_cancel_after_proposal() -> None:
    """Cancel a proposal run and preserve its public ledger."""

    from kazusa_ai_chatbot.coding_agent import (
        continue_coding_run,
        get_coding_run,
        start_coding_run,
    )

    gate_workspace, source_root, source_identity = _prepare_gate_workspace(GATE_04)
    source_hashes_before = _hash_source_files(source_root)

    start_response = await start_coding_run(
        _start_request(
            gate=GATE_04,
            gate_workspace=gate_workspace,
            source_root=source_root,
        )
    )
    response = await continue_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": start_response["run_id"],
        "action": "cancel",
        "reason": "Operator rejected the proposed change during review.",
    })
    reloaded_response = await get_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": response["run_id"],
    })

    source_hashes_after = _hash_source_files(source_root)
    raw_evidence = _raw_evidence(
        gate=GATE_04,
        source_root=source_root,
        gate_workspace=gate_workspace,
        source_identity=source_identity,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        response=response,
        reloaded_response=reloaded_response,
        start_response=start_response,
    )
    _write_json(_raw_evidence_path(GATE_04), raw_evidence)

    assert start_response["status"] == "awaiting_approval"
    _assert_common_run_response(
        gate=GATE_04,
        response=response,
        source_root=source_root,
        gate_workspace=gate_workspace,
    )
    assert response["status"] == "cancelled"
    assert response["patch_artifacts"]
    assert not response["apply_attempts"]
    assert not response["execution_attempts"]
    assert reloaded_response["status"] == "cancelled"
    assert raw_evidence["source_tree_unchanged"] is True


async def test_coding_run_live_gate_05_seeded_repair_attempt_ledger() -> None:
    """Run a seeded hard repair and inspect persisted attempt history."""

    from kazusa_ai_chatbot.coding_agent import get_coding_run, start_coding_run

    gate_workspace, source_root, source_identity = _prepare_gate_workspace(GATE_05)
    source_hashes_before = _hash_source_files(source_root)
    initial_patch_artifacts = _compile_initial_patch_artifacts(
        gate=GATE_05,
        source_root=source_root,
    )

    response = await start_coding_run(
        _start_request(
            gate=GATE_05,
            gate_workspace=gate_workspace,
            source_root=source_root,
            approval=_approval(GATE_05["gate_id"]),
            expected_source_identity=source_identity,
            initial_patch_artifacts=initial_patch_artifacts,
        )
    )
    reloaded_response = await get_coding_run({
        "workspace_root": str(gate_workspace),
        "run_id": response["run_id"],
    })

    source_hashes_after = _hash_source_files(source_root)
    raw_evidence = _raw_evidence(
        gate=GATE_05,
        source_root=source_root,
        gate_workspace=gate_workspace,
        source_identity=source_identity,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        response=response,
        reloaded_response=reloaded_response,
        initial_patch_artifact_count=len(initial_patch_artifacts),
    )
    _write_json(_raw_evidence_path(GATE_05), raw_evidence)

    _assert_common_run_response(
        gate=GATE_05,
        response=response,
        source_root=source_root,
        gate_workspace=gate_workspace,
    )
    assert response["status"] == "completed"
    assert len(response["attempts"]) >= 2
    _assert_expected_changed_paths(gate=GATE_05, response=response)
    _assert_final_execution_succeeded(response)
    _assert_protected_paths_not_changed(gate=GATE_05, response=response)
    assert reloaded_response["attempts"]
    assert raw_evidence["source_tree_unchanged"] is True


def _prepare_gate_workspace(
    gate: CodingRunGate,
) -> tuple[Path, Path, dict[str, object]]:
    """Create a fresh source checkout and run workspace for one gate."""

    gate_workspace = _reset_gate_workspace(gate["gate_id"])
    source_root = gate_workspace / "source"
    _write_source_files(source_root, gate["source_files"])
    source_identity = _initialize_fixture_git_checkout(
        source_root,
        gate["gate_id"],
    )
    prepared = (gate_workspace, source_root, source_identity)
    return prepared


def _reset_gate_workspace(gate_id: str) -> Path:
    """Create a clean managed workspace for one live gate run."""

    workspace_root = WORKSPACE_ROOT.resolve()
    gate_workspace = (WORKSPACE_ROOT / gate_id).resolve()
    if not gate_workspace.is_relative_to(workspace_root):
        raise AssertionError("Gate workspace escaped the managed workspace root.")

    if gate_workspace.exists():
        shutil.rmtree(gate_workspace, onerror=_retry_remove_readonly)
    gate_workspace.mkdir(parents=True, exist_ok=True)

    return gate_workspace


def _write_source_files(source_root: Path, source_files: dict[str, str]) -> None:
    """Write fixture source files for a live gate."""

    for relative_path, content in source_files.items():
        file_path = source_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8", newline="\n")


def _initialize_fixture_git_checkout(
    source_root: Path,
    gate_id: str,
) -> dict[str, object]:
    """Make a copied fixture satisfy the local checkout source contract."""

    commands = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "coding-agent-run@example.invalid"],
        ["git", "config", "user.name", "Coding Agent Run Gate"],
        ["git", "config", "core.autocrlf", "false"],
        ["git", "config", "core.eol", "lf"],
        [
            "git",
            "remote",
            "add",
            "origin",
            f"https://github.com/kazusa-fixtures/{gate_id}",
        ],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture baseline"],
    ]
    for command in commands:
        subprocess.run(
            command,
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    source_identity = {
        "provider": "github",
        "owner": "kazusa-fixtures",
        "repo": gate_id,
        "current_commit": commit_result.stdout.strip(),
        "dirty_state": "clean",
    }
    return source_identity


def _start_request(
    *,
    gate: CodingRunGate,
    gate_workspace: Path,
    source_root: Path,
    approval: dict[str, object] | None = None,
    expected_source_identity: dict[str, object] | None = None,
    initial_patch_artifacts: list[PatchArtifact] | None = None,
) -> dict[str, object]:
    """Build a public run-start request for a prepared gate."""

    request: dict[str, object] = {
        "question": gate["instruction"],
        "objective_type": gate["objective_type"],
        "workspace_root": str(gate_workspace),
        "local_root_hint": str(source_root),
        "source_scope_hint": "repository",
        "preferred_language": "en",
        "max_answer_chars": MAX_ANSWER_CHARS,
        "max_artifact_chars": MAX_ARTIFACT_CHARS,
    }
    execution_specs = gate.get("execution_specs")
    if execution_specs:
        request["execution_specs"] = execution_specs
    if approval is not None:
        request["approval"] = approval
    if expected_source_identity is not None:
        request["expected_source_identity"] = expected_source_identity
    if initial_patch_artifacts is not None:
        request["initial_patch_artifacts"] = initial_patch_artifacts
        request["repair_attempt_limit"] = 1
    return request


def _compile_initial_patch_artifacts(
    *,
    gate: CodingRunGate,
    source_root: Path,
) -> list[PatchArtifact]:
    """Compile seed operations into reviewed patch artifacts for a run."""

    patch_operations = gate["initial_operations"]
    patch_artifacts, _, _, errors = compile_patch_operations(
        repo_root=source_root,
        patch_operations=patch_operations,
        max_files=MAX_FILES,
        max_diff_chars=MAX_DIFF_CHARS,
    )
    if errors:
        raise AssertionError(f"Seed patch compilation failed: {errors}")
    if not patch_artifacts:
        raise AssertionError("Seed patch compilation produced no artifacts.")
    return patch_artifacts


def _approval(gate_id: str) -> dict[str, object]:
    """Build trusted approval metadata for a managed apply copy."""

    approval = {
        "approved": True,
        "approved_by": "coding-run-live-gate",
        "approved_at": "2026-07-09T00:00:00Z",
        "approval_reason": f"Prepared live coding-run gate {gate_id}.",
    }
    return approval


def _retry_remove_readonly(
    function: object,
    path: str,
    exc_info: object,
) -> None:
    """Clear Windows read-only bits left by git object files and retry."""

    del exc_info
    if not callable(function):
        raise AssertionError(f"Cleanup callback was not callable for {path}.")
    os.chmod(path, 0o700)
    function(path)


def _hash_source_files(root: Path) -> dict[str, str]:
    """Hash non-git files in a source tree for non-mutation checks."""

    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path.startswith(".git/"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative_path] = digest
    return hashes


def _raw_evidence(
    *,
    gate: CodingRunGate,
    source_root: Path,
    gate_workspace: Path,
    source_identity: dict[str, object],
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str],
    response: dict[str, Any],
    reloaded_response: dict[str, Any],
    start_response: dict[str, Any] | None = None,
    initial_patch_artifact_count: int = 0,
) -> dict[str, object]:
    """Build raw run evidence for later human-authored review."""

    evidence = {
        "gate": _gate_trace_contract(gate),
        "model_routes": [
            "CODING_AGENT_PM_LLM",
            "CODING_AGENT_PROGRAMMER_LLM",
        ],
        "source_root": str(source_root),
        "workspace_root": str(gate_workspace),
        "source_identity": source_identity,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_tree_unchanged": source_hashes_before == source_hashes_after,
        "initial_patch_artifact_count": initial_patch_artifact_count,
        "start_response": start_response,
        "response": response,
        "reloaded_response": reloaded_response,
    }
    return evidence


def _gate_trace_contract(gate: CodingRunGate) -> dict[str, object]:
    """Return the gate contract without embedding full source text twice."""

    trace_contract = {
        "gate_id": gate["gate_id"],
        "title": gate["title"],
        "objective_type": gate["objective_type"],
        "instruction": gate["instruction"],
        "source_paths": sorted(gate["source_files"]),
        "execution_specs": gate.get("execution_specs", []),
        "expected_status_after_start": gate["expected_status_after_start"],
        "expected_terminal_status": gate["expected_terminal_status"],
        "expected_changed_paths": gate["expected_changed_paths"],
        "protected_source_paths": gate["protected_source_paths"],
        "behavior_rubric": gate["behavior_rubric"],
        "forbidden_failure_modes": gate["forbidden_failure_modes"],
    }
    return trace_contract


def _raw_evidence_path(gate: CodingRunGate) -> Path:
    path = TRACE_ROOT / f"{gate['gate_id']}_raw_evidence.json"
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write raw structured evidence for later human-authored review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(encoded_payload, encoding="utf-8")


def _assert_common_run_response(
    *,
    gate: CodingRunGate,
    response: dict[str, Any],
    source_root: Path,
    gate_workspace: Path,
) -> None:
    """Assert public run metadata required for every gate."""

    assert response
    assert response["run_id"]
    assert response["status"] == gate["expected_terminal_status"]
    assert response["events"]
    assert response["trace_summary"]
    assert isinstance(response["limitations"], list)
    _assert_public_response_is_sanitized(
        response=response,
        source_root=source_root,
        gate_workspace=gate_workspace,
    )


def _assert_expected_changed_paths(
    *,
    gate: CodingRunGate,
    response: dict[str, Any],
) -> None:
    """Require expected runtime paths in proposal or final change summaries."""

    changed_paths = _changed_paths(response)
    for expected_path in gate["expected_changed_paths"]:
        assert expected_path in changed_paths


def _changed_paths(response: dict[str, Any]) -> set[str]:
    changed_paths: set[str] = set()
    for field_name in ("changed_files", "final_changed_files"):
        value = response.get(field_name)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if isinstance(path, str):
                changed_paths.add(path)

    for artifact in response.get("patch_artifacts", []):
        if not isinstance(artifact, dict):
            continue
        files = artifact.get("files")
        if not isinstance(files, list):
            continue
        for path in files:
            if isinstance(path, str):
                changed_paths.add(path)
    return changed_paths


def _assert_final_execution_succeeded(response: dict[str, Any]) -> None:
    """Require every final verification spec to pass."""

    execution_attempts = response["execution_attempts"]
    assert execution_attempts
    final_execution = execution_attempts[-1]
    results = final_execution.get("results")
    if isinstance(results, list):
        assert results
        for execution_result in results:
            assert execution_result["status"] == "succeeded"
        return
    assert final_execution["status"] == "succeeded"


def _assert_protected_paths_not_changed(
    *,
    gate: CodingRunGate,
    response: dict[str, Any],
) -> None:
    """Prevent run completion by changing protected verification sources."""

    changed_paths = _changed_paths(response)
    for protected_path in gate["protected_source_paths"]:
        assert protected_path not in changed_paths


def _assert_public_response_is_sanitized(
    *,
    response: dict[str, Any],
    source_root: Path,
    gate_workspace: Path,
) -> None:
    """Require public metadata to hide managed local roots."""

    raw_response = json.dumps(response, ensure_ascii=False)
    assert str(source_root.resolve()) not in raw_response
    assert str(gate_workspace.resolve()) not in raw_response
