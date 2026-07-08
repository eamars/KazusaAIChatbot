"""Live LLM gates for controlled verify-and-repair behavior."""

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

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_verify_repair")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_verify_repair_workspace")
MAX_ANSWER_CHARS = 6000
MAX_ARTIFACT_CHARS = 70000
MAX_DIFF_CHARS = 70000
MAX_FILES = 8
REPAIR_FEEDBACK_CHAR_LIMIT = 5000


class VerifyRepairGate(TypedDict):
    """One live verify-and-repair gate contract."""

    gate_id: str
    title: str
    instruction: str
    source_files: dict[str, str]
    initial_operations: list[PatchOperation]
    execution_specs: list[dict[str, object]]
    repair_attempt_limit: int
    expected_repaired_paths: list[str]
    protected_verification_paths: list[str]
    behavior_rubric: list[str]
    forbidden_failure_modes: list[str]


GATE_01: VerifyRepairGate = {
    "gate_id": "verify_repair_gate_01_median_boundary",
    "title": "Single-file median boundary repair",
    "instruction": (
        "Repair the statistics helper so median([]) raises ValueError and "
        "median with an even number of values returns the average of the two "
        "middle values. Do not modify the provided verification tests."
    ),
    "source_files": {
        "stats_tools.py": (
            "def median(values):\n"
            "    ordered = sorted(values)\n"
            "    middle = len(ordered) // 2\n"
            "    return ordered[middle]\n"
        ),
        "tests/test_stats_tools.py": (
            "import pytest\n\n"
            "from stats_tools import median\n\n\n"
            "def test_median_odd_length():\n"
            "    assert median([5, 1, 3]) == 3\n\n\n"
            "def test_median_even_length_averages_middle_pair():\n"
            "    assert median([10, 2, 4, 8]) == 6\n\n\n"
            "def test_median_rejects_empty_input():\n"
            "    with pytest.raises(ValueError):\n"
            "        median([])\n"
        ),
    },
    "initial_operations": [{
        "operation_id": "seed-empty-check-only",
        "kind": "replace_file_small",
        "path": "stats_tools.py",
        "content": (
            "def median(values):\n"
            "    ordered = sorted(values)\n"
            "    if not ordered:\n"
            "        raise ValueError('median requires at least one value')\n"
            "    middle = len(ordered) // 2\n"
            "    return ordered[middle]\n"
        ),
        "summary": "Seed patch adds empty-input handling but misses even lengths.",
    }],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_stats_tools.py"],
        "timeout_seconds": 20,
    }],
    "repair_attempt_limit": 1,
    "expected_repaired_paths": ["stats_tools.py"],
    "protected_verification_paths": ["tests/test_stats_tools.py"],
    "behavior_rubric": [
        "The first execution attempt fails on the even-length median contract.",
        "Repair feedback preserves the failing test name and assertion summary.",
        "The repaired attempt changes stats_tools.py and passes the same tests.",
    ],
    "forbidden_failure_modes": [
        "The repair edits tests instead of source behavior.",
        "The final response exposes absolute managed workspace paths.",
        "The repair loop repeats the seed patch without addressing even lengths.",
    ],
}

GATE_02: VerifyRepairGate = {
    "gate_id": "verify_repair_gate_02_cli_flag_handoff",
    "title": "Small multi-file CLI handoff repair",
    "instruction": (
        "Repair the word-count utility so --ignore-case is exposed by the CLI "
        "and passed into the counter. Core counting should keep current default "
        "case-sensitive behavior. Do not modify the provided verification tests."
    ),
    "source_files": {
        "wordcount/__init__.py": "",
        "wordcount/counter.py": (
            "def count_words(text):\n"
            "    counts = {}\n"
            "    for word in text.split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
        ),
        "wordcount/cli.py": (
            "import argparse\n\n"
            "from .counter import count_words\n\n\n"
            "def build_parser():\n"
            "    parser = argparse.ArgumentParser(prog='wordcount')\n"
            "    parser.add_argument('text')\n"
            "    return parser\n\n\n"
            "def main(argv=None):\n"
            "    args = build_parser().parse_args(argv)\n"
            "    counts = count_words(args.text)\n"
            "    for word, count in sorted(counts.items()):\n"
            "        print(f'{word}:{count}')\n"
        ),
        "tests/test_counter.py": (
            "from wordcount.counter import count_words\n\n\n"
            "def test_default_counting_is_case_sensitive():\n"
            "    assert count_words('Cat cat') == {'Cat': 1, 'cat': 1}\n\n\n"
            "def test_ignore_case_core_counting():\n"
            "    assert count_words('Cat cat DOG', ignore_case=True) == {\n"
            "        'cat': 2,\n"
            "        'dog': 1,\n"
            "    }\n"
        ),
        "tests/test_cli.py": (
            "from wordcount.cli import main\n\n\n"
            "def test_cli_ignore_case_flag(capsys):\n"
            "    main(['--ignore-case', 'Cat cat DOG'])\n"
            "    output = capsys.readouterr().out.splitlines()\n"
            "    assert output == ['cat:2', 'dog:1']\n"
        ),
    },
    "initial_operations": [{
        "operation_id": "seed-counter-only",
        "kind": "replace_file_small",
        "path": "wordcount/counter.py",
        "content": (
            "def count_words(text, *, ignore_case=False):\n"
            "    if ignore_case:\n"
            "        text = text.lower()\n"
            "    counts = {}\n"
            "    for word in text.split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
        ),
        "summary": "Seed patch updates the counter but omits CLI flag wiring.",
    }],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_counter.py", "tests/test_cli.py"],
        "timeout_seconds": 25,
    }],
    "repair_attempt_limit": 1,
    "expected_repaired_paths": ["wordcount/counter.py", "wordcount/cli.py"],
    "protected_verification_paths": ["tests/test_counter.py", "tests/test_cli.py"],
    "behavior_rubric": [
        "The first execution attempt proves the core helper is insufficient.",
        "Repair feedback points the PM/programmer toward CLI flag ownership.",
        "The repaired attempt preserves default case-sensitive counting.",
    ],
    "forbidden_failure_modes": [
        "The repair removes the core ignore_case parameter.",
        "The repair changes tests to avoid argparse coverage.",
        "The final attempt uses a new replacement CLI outside wordcount/cli.py.",
    ],
}

GATE_03: VerifyRepairGate = {
    "gate_id": "verify_repair_gate_03_duplicate_anchor_parser",
    "title": "Parser edge-case repair",
    "instruction": (
        "Repair the Markdown anchor collector so duplicate headings receive "
        "GitHub-style numeric suffixes. The first duplicate keeps the base "
        "anchor, the second gets -1, the third gets -2, and punctuation should "
        "not create different anchors. Do not modify the verification tests."
    ),
    "source_files": {
        "mdanchors.py": (
            "import re\n\n\n"
            "def slugify(text):\n"
            "    lowered = text.strip().lower()\n"
            "    lowered = re.sub(r'[^a-z0-9 ]+', '', lowered)\n"
            "    return re.sub(r'\\s+', '-', lowered).strip('-')\n\n\n"
            "def collect_anchors(markdown):\n"
            "    anchors = []\n"
            "    for line in markdown.splitlines():\n"
            "        if not line.startswith('#'):\n"
            "            continue\n"
            "        title = line.lstrip('#').strip()\n"
            "        anchors.append(slugify(title))\n"
            "    return anchors\n"
        ),
        "tests/test_mdanchors.py": (
            "from mdanchors import collect_anchors\n\n\n"
            "def test_duplicate_headings_receive_suffixes():\n"
            "    markdown = '# Intro\\n## Intro\\n# Intro!\\n'\n"
            "    assert collect_anchors(markdown) == [\n"
            "        'intro',\n"
            "        'intro-1',\n"
            "        'intro-2',\n"
            "    ]\n\n\n"
            "def test_distinct_headings_keep_base_slug():\n"
            "    markdown = '# Install Guide\\n# Usage\\n'\n"
            "    assert collect_anchors(markdown) == [\n"
            "        'install-guide',\n"
            "        'usage',\n"
            "    ]\n"
        ),
    },
    "initial_operations": [{
        "operation_id": "seed-punctuation-only",
        "kind": "replace_file_small",
        "path": "mdanchors.py",
        "content": (
            "import re\n\n\n"
            "def slugify(text):\n"
            "    lowered = text.strip().lower()\n"
            "    lowered = re.sub(r'[^a-z0-9 ]+', '', lowered)\n"
            "    return re.sub(r'\\s+', '-', lowered).strip('-')\n\n\n"
            "def collect_anchors(markdown):\n"
            "    anchors = []\n"
            "    for line in markdown.splitlines():\n"
            "        if not line.startswith('#'):\n"
            "            continue\n"
            "        title = line.lstrip('#').strip()\n"
            "        anchors.append(slugify(title))\n"
            "    return anchors\n"
        ),
        "summary": "Seed patch normalizes punctuation but omits duplicate suffixes.",
    }],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_mdanchors.py"],
        "timeout_seconds": 20,
    }],
    "repair_attempt_limit": 1,
    "expected_repaired_paths": ["mdanchors.py"],
    "protected_verification_paths": ["tests/test_mdanchors.py"],
    "behavior_rubric": [
        "The first execution attempt fails on duplicate-anchor suffixing.",
        "The repair changes parser state rather than special-casing one string.",
        "The repaired attempt preserves punctuation normalization.",
    ],
    "forbidden_failure_modes": [
        "The repair hard-codes the fixture headings.",
        "The repair updates tests instead of collect_anchors.",
        "The final response omits a repaired patch artifact for mdanchors.py.",
    ],
}

GATE_04: VerifyRepairGate = {
    "gate_id": "verify_repair_gate_04_soft_delete_cross_layer",
    "title": "Cross-layer soft-delete repair",
    "instruction": (
        "Repair the task tracker so delete marks a task archived instead of "
        "removing it. Normal list and single-item lookup should hide archived "
        "tasks, while list(include_archived=True) should expose them. Do not "
        "modify the provided store or API verification tests."
    ),
    "source_files": {
        "tasks/__init__.py": "",
        "tasks/models.py": (
            "from dataclasses import dataclass\n\n\n"
            "@dataclass\n"
            "class Task:\n"
            "    task_id: int\n"
            "    title: str\n"
        ),
        "tasks/store.py": (
            "from .models import Task\n\n\n"
            "class TaskStore:\n"
            "    def __init__(self):\n"
            "        self._items = {}\n\n"
            "    def add(self, task):\n"
            "        self._items[task.task_id] = task\n\n"
            "    def get(self, task_id):\n"
            "        return self._items.get(task_id)\n\n"
            "    def list(self):\n"
            "        return list(self._items.values())\n\n"
            "    def delete(self, task_id):\n"
            "        if task_id in self._items:\n"
            "            del self._items[task_id]\n"
        ),
        "tasks/api.py": (
            "class TaskAPI:\n"
            "    def __init__(self, store):\n"
            "        self.store = store\n\n"
            "    def list_tasks(self):\n"
            "        return [self._serialize(task) for task in self.store.list()]\n\n"
            "    def get_task(self, task_id):\n"
            "        task = self.store.get(task_id)\n"
            "        if task is None:\n"
            "            return None\n"
            "        return self._serialize(task)\n\n"
            "    def delete_task(self, task_id):\n"
            "        self.store.delete(task_id)\n"
            "        return {'deleted': True}\n\n"
            "    def _serialize(self, task):\n"
            "        return {'id': task.task_id, 'title': task.title}\n"
        ),
        "tests/test_store.py": (
            "from tasks.models import Task\n"
            "from tasks.store import TaskStore\n\n\n"
            "def test_delete_archives_without_normal_visibility():\n"
            "    store = TaskStore()\n"
            "    store.add(Task(task_id=1, title='ship patch'))\n"
            "    store.delete(1)\n"
            "    assert store.get(1) is None\n"
            "    assert store.list() == []\n"
            "    archived = store.list(include_archived=True)\n"
            "    assert len(archived) == 1\n"
            "    assert archived[0].archived is True\n"
        ),
        "tests/test_api.py": (
            "from tasks.api import TaskAPI\n"
            "from tasks.models import Task\n"
            "from tasks.store import TaskStore\n\n\n"
            "def test_api_hides_archived_tasks_by_default():\n"
            "    store = TaskStore()\n"
            "    api = TaskAPI(store)\n"
            "    store.add(Task(task_id=1, title='document'))\n"
            "    api.delete_task(1)\n"
            "    assert api.get_task(1) is None\n"
            "    assert api.list_tasks() == []\n"
            "    assert api.list_tasks(include_archived=True) == [{\n"
            "        'id': 1,\n"
            "        'title': 'document',\n"
            "        'archived': True,\n"
            "    }]\n"
        ),
    },
    "initial_operations": [{
        "operation_id": "seed-model-archived-field",
        "kind": "replace_file_small",
        "path": "tasks/models.py",
        "content": (
            "from dataclasses import dataclass\n\n\n"
            "@dataclass\n"
            "class Task:\n"
            "    task_id: int\n"
            "    title: str\n"
            "    archived: bool = False\n"
        ),
        "summary": "Seed patch adds the field but leaves store/API hard-delete behavior.",
    }],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_store.py", "tests/test_api.py"],
        "timeout_seconds": 25,
    }],
    "repair_attempt_limit": 1,
    "expected_repaired_paths": ["tasks/models.py", "tasks/store.py", "tasks/api.py"],
    "protected_verification_paths": ["tests/test_store.py", "tests/test_api.py"],
    "behavior_rubric": [
        "The first execution attempt proves model-only repair is insufficient.",
        "The repair coordinates model, store, and API semantics.",
        "The repaired attempt keeps archived tasks hidden by default.",
    ],
    "forbidden_failure_modes": [
        "The repair keeps hard delete and only changes API serialization.",
        "The repair creates a compatibility alias around old delete semantics.",
        "The repair edits verification tests instead of source behavior.",
    ],
}

GATE_05: VerifyRepairGate = {
    "gate_id": "verify_repair_gate_05_fetch_cache_cli",
    "title": "Hard mixed fetch/cache/CLI repair",
    "instruction": (
        "Repair the inventory sync fetch workflow so vendor page fetches accept "
        "a timeout, retry once after TimeoutError, use a file-backed response "
        "cache when cache_dir is provided, support refresh_cache, and expose "
        "--cache-dir, --refresh-cache, and --timeout through the CLI. Use only "
        "the Python standard library and keep tests mocked."
    ),
    "source_files": {
        "README.md": (
            "# Inventory Sync\n\n"
            "Fetch one vendor inventory page and print it to stdout.\n"
        ),
        "inventory_sync/__init__.py": "",
        "inventory_sync/fetch.py": (
            "from urllib.request import urlopen\n\n\n"
            "def fetch_page(url):\n"
            "    with urlopen(url) as response:\n"
            "        return response.read().decode('utf-8')\n"
        ),
        "inventory_sync/cli.py": (
            "import argparse\n\n"
            "from .fetch import fetch_page\n\n\n"
            "def build_parser():\n"
            "    parser = argparse.ArgumentParser(prog='inventory-sync')\n"
            "    parser.add_argument('url')\n"
            "    return parser\n\n\n"
            "def main(argv=None):\n"
            "    args = build_parser().parse_args(argv)\n"
            "    print(fetch_page(args.url))\n"
        ),
        "tests/test_fetch.py": (
            "from pathlib import Path\n\n"
            "from inventory_sync import fetch\n\n\n"
            "class FakeResponse:\n"
            "    def __init__(self, text):\n"
            "        self.text = text\n\n"
            "    def __enter__(self):\n"
            "        return self\n\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n\n"
            "    def read(self):\n"
            "        return self.text.encode('utf-8')\n\n\n"
            "def test_fetch_uses_timeout_and_retries_after_timeout(monkeypatch):\n"
            "    calls = []\n\n"
            "    def fake_urlopen(url, timeout):\n"
            "        calls.append((url, timeout))\n"
            "        if len(calls) == 1:\n"
            "            raise TimeoutError('slow vendor')\n"
            "        return FakeResponse('ok')\n\n"
            "    monkeypatch.setattr(fetch, 'urlopen', fake_urlopen)\n"
            "    assert fetch.fetch_page('https://vendor.test/feed', timeout=3) == 'ok'\n"
            "    assert calls == [\n"
            "        ('https://vendor.test/feed', 3),\n"
            "        ('https://vendor.test/feed', 3),\n"
            "    ]\n\n\n"
            "def test_fetch_uses_cache_without_second_network_call(monkeypatch, tmp_path):\n"
            "    calls = []\n\n"
            "    def fake_urlopen(url, timeout):\n"
            "        calls.append((url, timeout))\n"
            "        return FakeResponse('cached body')\n\n"
            "    monkeypatch.setattr(fetch, 'urlopen', fake_urlopen)\n"
            "    first = fetch.fetch_page(\n"
            "        'https://vendor.test/feed',\n"
            "        timeout=5,\n"
            "        cache_dir=tmp_path,\n"
            "    )\n"
            "    second = fetch.fetch_page(\n"
            "        'https://vendor.test/feed',\n"
            "        timeout=5,\n"
            "        cache_dir=tmp_path,\n"
            "    )\n"
            "    assert first == 'cached body'\n"
            "    assert second == 'cached body'\n"
            "    assert len(calls) == 1\n"
            "    assert any(Path(tmp_path).iterdir())\n"
        ),
        "tests/test_cli.py": (
            "from inventory_sync import cli\n\n\n"
            "def test_cli_passes_cache_and_timeout_flags(monkeypatch, tmp_path, capsys):\n"
            "    captured = {}\n\n"
            "    def fake_fetch_page(url, *, timeout, cache_dir=None, refresh_cache=False):\n"
            "        captured['url'] = url\n"
            "        captured['timeout'] = timeout\n"
            "        captured['cache_dir'] = cache_dir\n"
            "        captured['refresh_cache'] = refresh_cache\n"
            "        return 'body'\n\n"
            "    monkeypatch.setattr(cli, 'fetch_page', fake_fetch_page)\n"
            "    cli.main([\n"
            "        '--cache-dir',\n"
            "        str(tmp_path),\n"
            "        '--refresh-cache',\n"
            "        '--timeout',\n"
            "        '7',\n"
            "        'https://vendor.test/feed',\n"
            "    ])\n"
            "    assert capsys.readouterr().out.strip() == 'body'\n"
            "    assert captured == {\n"
            "        'url': 'https://vendor.test/feed',\n"
            "        'timeout': 7,\n"
            "        'cache_dir': tmp_path,\n"
            "        'refresh_cache': True,\n"
            "    }\n"
        ),
    },
    "initial_operations": [{
        "operation_id": "seed-timeout-only",
        "kind": "replace_file_small",
        "path": "inventory_sync/fetch.py",
        "content": (
            "from urllib.request import urlopen\n\n\n"
            "def fetch_page(url, *, timeout=10):\n"
            "    with urlopen(url, timeout=timeout) as response:\n"
            "        return response.read().decode('utf-8')\n"
        ),
        "summary": "Seed patch adds timeout but omits retry, cache, and CLI flags.",
    }],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_fetch.py", "tests/test_cli.py"],
        "timeout_seconds": 35,
    }],
    "repair_attempt_limit": 2,
    "expected_repaired_paths": ["inventory_sync/fetch.py", "inventory_sync/cli.py"],
    "protected_verification_paths": ["tests/test_fetch.py", "tests/test_cli.py"],
    "behavior_rubric": [
        "The first execution attempt fails on retry/cache/CLI omissions.",
        "Repair feedback is redacted but preserves failing test names.",
        "The repair uses mocked tests and adds no real network calls.",
        "The final attempt updates fetch behavior and CLI flag wiring together.",
    ],
    "forbidden_failure_modes": [
        "The repair changes tests to avoid cache or timeout assertions.",
        "The repair performs a real network call during tests.",
        "The final response exposes absolute workspace paths or raw shell text.",
    ],
}

GATE_06: VerifyRepairGate = {
    "gate_id": "verify_repair_gate_06_release_feed_cache_cli",
    "title": "Release feed offline cache and CLI repair",
    "instruction": (
        "Repair the release feed loader so network fetches accept a timeout, "
        "retry once after TimeoutError, write a caller-provided cache file "
        "after a successful fetch, read that cache file when offline=True, "
        "and expose --cache-file, --offline, and --timeout through the CLI. "
        "Use only the Python standard library and keep network behavior "
        "mocked in tests."
    ),
    "source_files": {
        "release_feed/__init__.py": "",
        "release_feed/client.py": (
            "from urllib.request import urlopen\n\n\n"
            "def load_feed(url):\n"
            "    with urlopen(url) as response:\n"
            "        return response.read().decode('utf-8')\n"
        ),
        "release_feed/cli.py": (
            "import argparse\n\n"
            "from .client import load_feed\n\n\n"
            "def build_parser():\n"
            "    parser = argparse.ArgumentParser(prog='release-feed')\n"
            "    parser.add_argument('url')\n"
            "    return parser\n\n\n"
            "def main(argv=None):\n"
            "    args = build_parser().parse_args(argv)\n"
            "    print(load_feed(args.url))\n"
        ),
        "tests/test_client.py": (
            "from release_feed import client\n\n\n"
            "class FakeResponse:\n"
            "    def __init__(self, text):\n"
            "        self.text = text\n\n"
            "    def __enter__(self):\n"
            "        return self\n\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n\n"
            "    def read(self):\n"
            "        return self.text.encode('utf-8')\n\n\n"
            "def test_load_feed_uses_offline_cache_without_network(monkeypatch, tmp_path):\n"
            "    cache_file = tmp_path / 'feed.json'\n"
            "    cache_file.write_text('{\"version\": \"cached\"}', encoding='utf-8')\n\n"
            "    def fake_urlopen(url, timeout):\n"
            "        raise AssertionError('offline mode must not use network')\n\n"
            "    monkeypatch.setattr(client, 'urlopen', fake_urlopen)\n"
            "    result = client.load_feed(\n"
            "        'https://release.test/feed.json',\n"
            "        timeout=4,\n"
            "        cache_file=cache_file,\n"
            "        offline=True,\n"
            "    )\n"
            "    assert result == '{\"version\": \"cached\"}'\n\n\n"
            "def test_load_feed_retries_timeout_and_updates_cache(monkeypatch, tmp_path):\n"
            "    calls = []\n"
            "    cache_file = tmp_path / 'feed.json'\n\n"
            "    def fake_urlopen(url, timeout):\n"
            "        calls.append((url, timeout))\n"
            "        if len(calls) == 1:\n"
            "            raise TimeoutError('slow release server')\n"
            "        return FakeResponse('{\"version\": \"fresh\"}')\n\n"
            "    monkeypatch.setattr(client, 'urlopen', fake_urlopen)\n"
            "    result = client.load_feed(\n"
            "        'https://release.test/feed.json',\n"
            "        timeout=8,\n"
            "        cache_file=cache_file,\n"
            "    )\n"
            "    assert result == '{\"version\": \"fresh\"}'\n"
            "    assert cache_file.read_text(encoding='utf-8') == '{\"version\": \"fresh\"}'\n"
            "    assert calls == [\n"
            "        ('https://release.test/feed.json', 8),\n"
            "        ('https://release.test/feed.json', 8),\n"
            "    ]\n"
        ),
        "tests/test_cli.py": (
            "from release_feed import cli\n\n\n"
            "def test_cli_passes_cache_offline_and_timeout(monkeypatch, tmp_path, capsys):\n"
            "    captured = {}\n"
            "    cache_file = tmp_path / 'feed.json'\n\n"
            "    def fake_load_feed(url, *, timeout, cache_file=None, offline=False):\n"
            "        captured['url'] = url\n"
            "        captured['timeout'] = timeout\n"
            "        captured['cache_file'] = cache_file\n"
            "        captured['offline'] = offline\n"
            "        return '{\"version\": \"cached\"}'\n\n"
            "    monkeypatch.setattr(cli, 'load_feed', fake_load_feed)\n"
            "    cli.main([\n"
            "        '--cache-file',\n"
            "        str(cache_file),\n"
            "        '--offline',\n"
            "        '--timeout',\n"
            "        '9',\n"
            "        'https://release.test/feed.json',\n"
            "    ])\n"
            "    assert capsys.readouterr().out.strip() == '{\"version\": \"cached\"}'\n"
            "    assert captured == {\n"
            "        'url': 'https://release.test/feed.json',\n"
            "        'timeout': 9,\n"
            "        'cache_file': cache_file,\n"
            "        'offline': True,\n"
            "    }\n"
        ),
    },
    "initial_operations": [{
        "operation_id": "seed-release-timeout-only",
        "kind": "replace_file_small",
        "path": "release_feed/client.py",
        "content": (
            "from urllib.request import urlopen\n\n\n"
            "def load_feed(url, *, timeout=10):\n"
            "    with urlopen(url, timeout=timeout) as response:\n"
            "        return response.read().decode('utf-8')\n"
        ),
        "summary": (
            "Seed patch adds timeout but omits retry, cache-file handling, "
            "offline mode, and CLI flags."
        ),
    }],
    "execution_specs": [{
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": ["tests/test_client.py", "tests/test_cli.py"],
        "timeout_seconds": 35,
    }],
    "repair_attempt_limit": 2,
    "expected_repaired_paths": ["release_feed/client.py", "release_feed/cli.py"],
    "protected_verification_paths": ["tests/test_client.py", "tests/test_cli.py"],
    "behavior_rubric": [
        "The first execution attempt fails on mocked timeout/cache/CLI behavior.",
        "Repair feedback preserves failing test names without raw full output.",
        "The repair implements source behavior from existing mocked I/O patterns.",
        "The final attempt updates client behavior and CLI flag wiring together.",
    ],
    "forbidden_failure_modes": [
        "The repair edits tests instead of client or CLI source.",
        "The repair treats absent implementation helpers as a blocker.",
        "The repair performs real network I/O during tests.",
        "The final response exposes absolute workspace paths or raw shell text.",
    ],
}


async def test_verify_repair_live_gate_01_median_boundary() -> None:
    """Run the simple single-file repair gate."""

    response = await _run_verify_repair_gate(GATE_01)
    _assert_verify_repair_response(gate=GATE_01, response=response)


async def test_verify_repair_live_gate_02_cli_flag_handoff() -> None:
    """Run the small multi-file repair gate."""

    response = await _run_verify_repair_gate(GATE_02)
    _assert_verify_repair_response(gate=GATE_02, response=response)


async def test_verify_repair_live_gate_03_duplicate_anchor_parser() -> None:
    """Run the parser edge-case repair gate."""

    response = await _run_verify_repair_gate(GATE_03)
    _assert_verify_repair_response(gate=GATE_03, response=response)


async def test_verify_repair_live_gate_04_soft_delete_cross_layer() -> None:
    """Run the cross-layer behavior repair gate."""

    response = await _run_verify_repair_gate(GATE_04)
    _assert_verify_repair_response(gate=GATE_04, response=response)


async def test_verify_repair_live_gate_05_fetch_cache_cli() -> None:
    """Run the hard mixed fetch/cache/CLI repair gate."""

    response = await _run_verify_repair_gate(GATE_05)
    _assert_verify_repair_response(gate=GATE_05, response=response)


async def test_verify_repair_live_gate_06_release_feed_cache_cli() -> None:
    """Run the retained hard mocked-I/O repair gate."""

    response = await _run_verify_repair_gate(GATE_06)
    _assert_verify_repair_response(gate=GATE_06, response=response)


async def _run_verify_repair_gate(gate: VerifyRepairGate) -> dict[str, Any]:
    """Run one live repair gate and persist raw structured evidence."""

    from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change

    gate_id = gate["gate_id"]
    workspace_root = _reset_gate_workspace(gate_id)
    source_root = workspace_root / "source"
    _write_source_tree(source_root=source_root, files=gate["source_files"])
    source_identity = _initialize_fixture_git_checkout(source_root, gate_id)
    initial_patch_artifacts = _compile_initial_patch_artifacts(
        gate=gate,
        source_root=source_root,
    )

    before_hashes = _hash_source_files(source_root)
    request = _verify_repair_request(
        gate=gate,
        source_root=source_root,
        workspace_root=workspace_root,
        source_identity=source_identity,
        initial_patch_artifacts=initial_patch_artifacts,
    )
    response = await verify_and_repair_code_change(request)
    after_hashes = _hash_source_files(source_root)

    trace_payload = {
        "gate": _gate_trace_contract(gate),
        "model_routes": [
            "CODING_AGENT_PM_LLM",
            "CODING_AGENT_PROGRAMMER_LLM",
        ],
        "source_root": str(source_root),
        "workspace_root": str(workspace_root),
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_tree_unchanged": before_hashes == after_hashes,
        "initial_patch_artifacts": initial_patch_artifacts,
        "request": _request_trace_summary(request),
        "response": response,
        "human_review_required": True,
    }
    trace_path = TRACE_ROOT / f"{gate_id}_raw_evidence.json"
    _write_json(trace_path, trace_payload)

    print(f"gate_id={gate_id}")
    print(f"raw_evidence_path={trace_path}")
    print(f"source_tree_unchanged={before_hashes == after_hashes}")

    return response


def _verify_repair_request(
    *,
    gate: VerifyRepairGate,
    source_root: Path,
    workspace_root: Path,
    source_identity: dict[str, object],
    initial_patch_artifacts: list[PatchArtifact],
) -> dict[str, object]:
    """Build the trusted direct repair request for one live gate."""

    request: dict[str, object] = {
        "question": gate["instruction"],
        "local_root_hint": str(source_root),
        "source_scope_hint": "directory",
        "workspace_root": str(workspace_root),
        "session_id": gate["gate_id"],
        "preferred_language": "English",
        "max_answer_chars": MAX_ANSWER_CHARS,
        "max_artifact_chars": MAX_ARTIFACT_CHARS,
        "approval": _approval(gate["gate_id"]),
        "execution_specs": gate["execution_specs"],
        "repair_attempt_limit": gate["repair_attempt_limit"],
        "max_repair_feedback_chars": REPAIR_FEEDBACK_CHAR_LIMIT,
        "initial_patch_artifacts": initial_patch_artifacts,
        "expected_source_identity": source_identity,
    }
    return request


def _compile_initial_patch_artifacts(
    *,
    gate: VerifyRepairGate,
    source_root: Path,
) -> list[PatchArtifact]:
    """Compile the seeded failing patch used to enter the repair path."""

    patch_artifacts, _, _, errors = compile_patch_operations(
        repo_root=source_root,
        patch_operations=gate["initial_operations"],
        max_files=MAX_FILES,
        max_diff_chars=MAX_DIFF_CHARS,
    )
    assert errors == []
    assert patch_artifacts
    return patch_artifacts


def _write_source_tree(*, source_root: Path, files: dict[str, str]) -> None:
    """Write the source fixture files for one live gate."""

    source_root.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = source_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def _initialize_fixture_git_checkout(
    source_root: Path,
    gate_id: str,
) -> dict[str, object]:
    """Make a copied fixture satisfy the local checkout source contract."""

    commands = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "coding-agent-gate@example.invalid"],
        ["git", "config", "user.name", "Coding Agent Gate"],
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


def _approval(gate_id: str) -> dict[str, object]:
    """Build trusted approval metadata for the managed apply copy."""

    approval = {
        "approved": True,
        "approved_by": "verify-repair-live-gate",
        "approved_at": "2026-07-08T00:00:00Z",
        "approval_reason": f"Prepared live repair gate {gate_id}.",
    }
    return approval


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write raw structured evidence for later human-authored review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(encoded_payload, encoding="utf-8")


def _gate_trace_contract(gate: VerifyRepairGate) -> dict[str, object]:
    """Return the gate contract without embedding full source text twice."""

    trace_contract = {
        "gate_id": gate["gate_id"],
        "title": gate["title"],
        "instruction": gate["instruction"],
        "source_paths": sorted(gate["source_files"]),
        "execution_specs": gate["execution_specs"],
        "repair_attempt_limit": gate["repair_attempt_limit"],
        "expected_repaired_paths": gate["expected_repaired_paths"],
        "protected_verification_paths": gate["protected_verification_paths"],
        "behavior_rubric": gate["behavior_rubric"],
        "forbidden_failure_modes": gate["forbidden_failure_modes"],
    }
    return trace_contract


def _request_trace_summary(request: dict[str, object]) -> dict[str, object]:
    """Keep trace input readable without duplicating full patch diffs."""

    initial_patch_artifacts = request["initial_patch_artifacts"]
    assert isinstance(initial_patch_artifacts, list)
    summary = {
        "question": request["question"],
        "source_scope_hint": request["source_scope_hint"],
        "execution_specs": request["execution_specs"],
        "repair_attempt_limit": request["repair_attempt_limit"],
        "initial_patch_artifact_count": len(initial_patch_artifacts),
    }
    return summary


def _assert_verify_repair_response(
    *,
    gate: VerifyRepairGate,
    response: dict[str, Any],
) -> None:
    """Assert structural evidence required for repair gate review."""

    assert response
    assert response["status"] == "succeeded"
    assert response["answer_text"]
    assert response["attempts"]
    assert len(response["attempts"]) >= 2
    assert response["final_patch_artifacts"]
    assert response["final_changed_files"]
    assert response["final_apply"]
    assert response["final_execution"]

    _assert_first_attempt_failed(response)
    _assert_final_execution_succeeded(response)
    _assert_fresh_apply_workspace_per_attempt(response)
    _assert_expected_repaired_paths(gate=gate, response=response)
    _assert_verification_tests_not_modified(gate=gate, response=response)
    _assert_public_response_is_sanitized(gate=gate, response=response)
    _assert_source_tree_unchanged(gate=gate)


def _assert_first_attempt_failed(response: dict[str, Any]) -> None:
    """Require the prepared seed to exercise the repair path."""

    first_attempt = response["attempts"][0]
    execution_statuses = first_attempt["execution_statuses"]
    assert any(
        status in {"failed", "timed_out"}
        for status in execution_statuses
    )


def _assert_final_execution_succeeded(response: dict[str, Any]) -> None:
    """Require every final verification spec to pass in one attempt."""

    for execution_result in response["final_execution"]:
        assert execution_result["status"] == "succeeded"


def _assert_fresh_apply_workspace_per_attempt(response: dict[str, Any]) -> None:
    """Require repair attempts to use fresh managed apply packages."""

    apply_package_ids: list[str] = []
    for attempt in response["attempts"]:
        apply_package_id = attempt.get("apply_package_id")
        if apply_package_id is None:
            continue
        assert isinstance(apply_package_id, str)
        assert apply_package_id
        apply_package_ids.append(apply_package_id)

    assert len(apply_package_ids) >= 2
    assert len(set(apply_package_ids)) == len(apply_package_ids)


def _assert_expected_repaired_paths(
    *,
    gate: VerifyRepairGate,
    response: dict[str, Any],
) -> None:
    """Require final repaired artifacts for the expected source paths."""

    changed_paths = [
        str(item["path"])
        for item in response["final_changed_files"]
    ]
    for expected_path in gate["expected_repaired_paths"]:
        assert any(expected_path == path for path in changed_paths)


def _assert_verification_tests_not_modified(
    *,
    gate: VerifyRepairGate,
    response: dict[str, Any],
) -> None:
    """Prevent a repair from passing by editing the gate tests."""

    changed_paths = [
        str(item["path"])
        for item in response["final_changed_files"]
    ]
    for protected_path in gate["protected_verification_paths"]:
        assert protected_path not in changed_paths


def _assert_public_response_is_sanitized(
    *,
    gate: VerifyRepairGate,
    response: dict[str, Any],
) -> None:
    """Require public metadata to hide managed local roots."""

    copied_source_root = WORKSPACE_ROOT / gate["gate_id"] / "source"
    gate_workspace = WORKSPACE_ROOT / gate["gate_id"]
    raw_response = json.dumps(response, ensure_ascii=False)
    assert str(copied_source_root.resolve()) not in raw_response
    assert str(gate_workspace.resolve()) not in raw_response


def _assert_source_tree_unchanged(gate: VerifyRepairGate) -> None:
    """Require verify/repair to leave the source checkout unchanged."""

    raw_evidence_path = TRACE_ROOT / f"{gate['gate_id']}_raw_evidence.json"
    raw_evidence = json.loads(raw_evidence_path.read_text(encoding="utf-8"))
    assert raw_evidence["source_tree_unchanged"] is True
