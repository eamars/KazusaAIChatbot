import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = result.stdout.strip()
    return output


def _make_reading_git_checkout(tmp_path: Path) -> Path:
    repo_root = tmp_path / "direct_repo"
    (repo_root / "src" / "app").mkdir(parents=True)
    (repo_root / "src" / "app" / "image_pipeline.py").write_text(
        "\n".join(
            [
                "VISION_DESCRIPTOR_LLM = 'vision-descriptor'",
                "",
                "",
                "def read_image(attachment: dict) -> dict:",
                "    payload = attachment['base64_data']",
                "    return {",
                "        'payload': payload,",
                "        'image_observation': VISION_DESCRIPTOR_LLM,",
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text(
        "Images use base64_data and image_observation.\n",
        encoding="utf-8",
    )
    _run_git(["init"], repo_root)
    _run_git(["config", "user.email", "test@example.com"], repo_root)
    _run_git(["config", "user.name", "Test User"], repo_root)
    _run_git(["add", "."], repo_root)
    _run_git(["commit", "-m", "initial"], repo_root)
    _run_git(
        [
            "remote",
            "add",
            "origin",
            "https://github.com/fixture/reader.git",
        ],
        repo_root,
    )
    return repo_root


def _repository(repo_root: Path, dirty_state: str = "clean") -> dict[str, Any]:
    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "reader",
        "source_url": "https://github.com/fixture/reader",
        "requested_ref": "main",
        "resolved_ref": "main",
        "current_commit": "b" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(repo_root.parent / "workspace"),
        "cache_key": "github-fixture-reader-main",
        "dirty_state": dirty_state,
    }
    return repository


def _repository_scope() -> dict[str, Any]:
    scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://github/fixture/reader",
        "requested_ref": "main",
        "interpretation": "entire repository",
    }
    return scope


def _successful_fetching_result(
    repo_root: Path,
    dirty_state: str = "clean",
) -> dict[str, Any]:
    result = {
        "status": "succeeded",
        "message": "resolved",
        "repository": _repository(repo_root, dirty_state=dirty_state),
        "source_scope": _repository_scope(),
        "limitations": ["fetch limitation"],
        "trace_summary": ["fetch:resolved"],
    }
    return result


def _successful_reading_result() -> dict[str, Any]:
    result = {
        "status": "succeeded",
        "answer_text": "ImagePipeline describes attachments.",
        "evidence": [
            {
                "path": "src/app/image_pipeline.py",
                "line_start": 1,
                "line_end": 8,
                "symbol_or_topic": "ImagePipeline",
                "excerpt": "class ImagePipeline:",
                "reason": "Defines the image reader.",
            }
        ],
        "limitations": ["reading limitation"],
        "trace_summary": ["reading:evidence"],
    }
    return result


def _install_fake_subagents(
    monkeypatch: pytest.MonkeyPatch,
    fetching_result: dict[str, Any],
    reading_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading

    calls: dict[str, Any] = {"fetching": [], "reading": []}

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["fetching"].append(request)
        return fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["reading"].append(request)
        if reading_result is None:
            raise AssertionError("code_reading.run should not be called")
        return reading_result

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    return calls


async def test_answer_code_question_short_circuits_failed_source_fetching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question

    fetching_result = {
        "status": "needs_user_input",
        "message": "Which repository should I read?",
        "repository": None,
        "source_scope": None,
        "limitations": ["fetch needs a repository"],
        "trace_summary": ["fetch:needs_user_input"],
    }
    calls = _install_fake_subagents(monkeypatch, fetching_result)

    response = await answer_code_question(
        {
            "question": "How does this project read images?",
            "workspace_root": "C:/tmp/workspace",
        }
    )

    assert response["status"] == "needs_user_input"
    assert response["answer_text"] == ""
    assert response["repository"] is None
    assert response["source_scope"] is None
    assert response["evidence"] == []
    assert response["limitations"] == ["fetch needs a repository"]
    assert "fetch:needs_user_input" in response["trace_summary"]
    assert calls["reading"] == []


async def test_answer_code_question_passes_public_fetching_fields_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question

    fetching_result = _successful_fetching_result(tmp_path / "repo")
    calls = _install_fake_subagents(
        monkeypatch,
        fetching_result,
        reading_result=_successful_reading_result(),
    )
    request = {
        "question": (
            "Read https://github.com/fixture/reader and explain images."
        ),
        "source_url": "https://github.com/fixture/reader",
        "repo_url": "https://github.com/fixture/reader.git",
        "repo_hint": "fixture/reader",
        "local_root_hint": str(tmp_path / "repo"),
        "local_path_hint": str(tmp_path / "repo" / "src" / "app"),
        "requested_ref": "main",
        "source_scope_hint": "directory",
        "workspace_root": str(tmp_path / "workspace"),
        "preferred_language": "English",
        "max_answer_chars": 1200,
    }

    response = await answer_code_question(request)

    assert response["status"] == "succeeded"
    assert calls["fetching"] == [request]
    reading_request = calls["reading"][0]
    assert reading_request["question"] == request["question"]
    assert reading_request["repository"] == fetching_result["repository"]
    assert reading_request["source_scope"] == fetching_result["source_scope"]
    assert reading_request["preferred_language"] == "English"
    assert reading_request["max_answer_chars"] == 1200


async def test_answer_code_question_sanitizes_repository_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question

    repo_root = tmp_path / "repo"
    fetching_result = _successful_fetching_result(repo_root)
    calls = _install_fake_subagents(
        monkeypatch,
        fetching_result,
        reading_result=_successful_reading_result(),
    )

    response = await answer_code_question(
        {
            "question": "How does this project read images?",
            "local_root_hint": str(repo_root),
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert response["repository"] == {
        "provider": "github",
        "owner": "fixture",
        "repo": "reader",
        "source_url": "https://github.com/fixture/reader",
        "requested_ref": "main",
        "resolved_ref": "main",
        "current_commit": "b" * 40,
        "default_branch": "main",
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "dirty_state": "clean",
    }
    assert response["source_scope"] == fetching_result["source_scope"]
    assert "local_root" not in repr(response)
    assert "workspace_root" not in repr(response)
    assert "cache_key" not in repr(response)
    assert str(repo_root) not in repr(response)
    assert str(tmp_path / "workspace") not in repr(response)
    assert calls["reading"]


async def test_answer_code_question_sanitizes_inline_repository_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question

    inline_root = tmp_path / 'inline_bundle'
    inline_root.mkdir()
    (inline_root / 'snippet.py').write_text(
        'def solve():\n'
        '    return 1\n',
        encoding='utf-8',
    )
    fetching_result = {
        'status': 'succeeded',
        'message': 'resolved',
        'repository': {
            'provider': 'inline',
            'owner': 'inline',
            'repo': 'bundle-abc123',
            'source_url': 'inline://accepted-task/bundle-abc123',
            'requested_ref': None,
            'resolved_ref': 'inline',
            'current_commit': 'inline-sha256:' + 'a' * 64,
            'default_branch': '',
            'local_root': str(inline_root),
            'storage_kind': 'managed_inline_bundle',
            'managed_checkout': True,
            'workspace_root': str(tmp_path / 'workspace'),
            'cache_key': None,
            'dirty_state': 'clean',
        },
        'source_scope': {
            'kind': 'file',
            'repo_relative_path': 'snippet.py',
            'source_url': 'inline://accepted-task/bundle-abc123/snippet.py',
            'requested_ref': None,
            'interpretation': 'managed inline source fragment',
        },
        'limitations': [],
        'trace_summary': ['managed_inline:materialized:1'],
    }
    calls = _install_fake_subagents(
        monkeypatch,
        fetching_result,
        reading_result=_successful_reading_result(),
    )

    response = await answer_code_question({
        'question': 'Please review the inline snippet.',
        'workspace_root': str(tmp_path / 'workspace'),
    })

    assert response['status'] == 'succeeded'
    assert response['repository'] == {
        'provider': 'inline',
        'owner': 'inline',
        'repo': 'bundle-abc123',
        'source_url': 'inline://accepted-task/bundle-abc123',
        'requested_ref': None,
        'resolved_ref': 'inline',
        'current_commit': 'inline-sha256:' + 'a' * 64,
        'default_branch': '',
        'storage_kind': 'managed_inline_bundle',
        'managed_checkout': True,
        'dirty_state': 'clean',
    }
    assert calls['reading'][0]['repository']['local_root'] == str(inline_root)
    assert str(inline_root) not in repr(response)
    assert str(tmp_path / 'workspace') not in repr(response)
    assert 'cache_key' not in repr(response)


async def test_answer_code_question_carries_ordered_limitations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question

    repo_root = tmp_path / "repo"
    fetching_result = _successful_fetching_result(repo_root, dirty_state="dirty")
    _install_fake_subagents(
        monkeypatch,
        fetching_result,
        reading_result=_successful_reading_result(),
    )

    response = await answer_code_question(
        {
            "question": "How does this project read images?",
            "local_root_hint": str(repo_root),
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert response["limitations"][0] == "fetch limitation"
    assert (
        "Existing local checkout is dirty; evidence may include "
        "uncommitted local changes."
    ) in response["limitations"]
    assert response["limitations"][-1] == "reading limitation"


async def test_question_only_embedded_github_url_is_handed_to_fetching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question

    fetching_result = _successful_fetching_result(tmp_path / "repo")
    calls = _install_fake_subagents(
        monkeypatch,
        fetching_result,
        reading_result=_successful_reading_result(),
    )
    question = (
        "Explain [fixture/reader](https://github.com/fixture/reader) image "
        "reading."
    )

    response = await answer_code_question(
        {
            "question": question,
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert calls["fetching"] == [
        {
            "question": question,
            "workspace_root": str(tmp_path / "workspace"),
        }
    ]


async def test_answer_code_question_reads_real_local_checkout_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question
    from kazusa_ai_chatbot.coding_agent import code_reading

    repo_root = _make_reading_git_checkout(tmp_path)
    reading_calls: list[dict[str, Any]] = []

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        reading_calls.append(request)
        result = _successful_reading_result()
        return result

    monkeypatch.setattr(code_reading, "run", fake_reading_run)

    response = await answer_code_question(
        {
            "question": "How does this project read images?",
            "local_root_hint": str(repo_root),
            "workspace_root": str(tmp_path / "workspace"),
            "preferred_language": "English",
            "max_answer_chars": 1200,
        }
    )

    assert response["status"] == "succeeded"
    assert response["repository"] is not None
    assert response["repository"]["owner"] == "fixture"
    assert response["repository"]["repo"] == "reader"
    assert "local_root" not in repr(response)
    assert "workspace_root" not in repr(response)
    assert "cache_key" not in repr(response)
    assert response["source_scope"] is not None
    assert response["source_scope"]["kind"] == "repository"
    assert reading_calls
    reading_request = reading_calls[0]
    assert reading_request["repository"]["local_root"] == str(repo_root)
    assert reading_request["source_scope"]["kind"] == "repository"
    assert {row["path"] for row in response["evidence"]} & {
        "src/app/image_pipeline.py",
        "README.md",
    }


async def test_propose_code_change_resolves_generated_readback_through_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    writing_calls: list[dict[str, Any]] = []
    reading_calls: list[dict[str, Any]] = []

    async def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        writing_calls.append(request)
        if len(writing_calls) == 1:
            readback_root = tmp_path / "readback"
            readback_root.mkdir()
            (readback_root / "runtime.py").write_text(
                "def parse_rows(path: str) -> list[dict[str, str]]:\n"
                "    return []\n",
                encoding="utf-8",
            )
            return {
                "status": "need_reading",
                "mode": "create_new_project",
                "answer_text": "Need generated source readback.",
                "patch_artifacts": [],
                "created_files": [],
                "changed_files": [],
                "reading_requests": [
                    {
                        "request_id": "runtime_readback",
                        "task": "Read parse_rows signature and row shape.",
                        "reason": "Tests must match source.",
                        "target_artifacts": ["runtime.py"],
                    }
                ],
                "reading_source": {
                    "repository": {
                        "provider": "github",
                        "owner": "managed",
                        "repo": "generated-artifacts",
                        "source_url": "local://readback",
                        "requested_ref": None,
                        "resolved_ref": "generated",
                        "current_commit": "generated-sha256:test",
                        "default_branch": "generated",
                        "local_root": str(readback_root),
                        "storage_kind": "managed_download",
                        "managed_checkout": True,
                        "workspace_root": str(tmp_path / "workspace"),
                        "cache_key": None,
                        "dirty_state": "clean",
                    },
                    "source_scope": {
                        "kind": "repository",
                        "repo_relative_path": None,
                        "source_url": "local://readback",
                        "requested_ref": None,
                        "interpretation": "Generated readback.",
                    },
                },
                "external_evidence_requests": [],
                "external_evidence": [],
                "validation": {
                    "status": "failed",
                    "parsed": False,
                    "sandbox_applied": False,
                    "errors": [],
                    "warnings": [],
                    "files": [],
                },
                "session": {"session_id": "s1", "public_handle": "h1"},
                "limitations": [],
                "trace_summary": ["writing_pm:decision status=need_reading"],
            }
        assert request["supervisor_facts"][0]["request_id"] == "runtime_readback"
        return {
            "status": "succeeded",
            "mode": "create_new_project",
            "answer_text": "Generated source and tests.",
            "patch_artifacts": [],
            "created_files": [{"path": "runtime.py", "role": "source"}],
            "changed_files": [],
            "external_evidence_requests": [],
            "external_evidence": [],
            "validation": {
                "status": "succeeded",
                "parsed": True,
                "sandbox_applied": True,
                "errors": [],
                "warnings": [],
                "files": ["runtime.py"],
            },
            "session": {"session_id": "s1", "public_handle": "h1"},
            "limitations": [],
            "trace_summary": ["writing_pm:complete"],
        }

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        reading_calls.append(request)
        return {
            "status": "succeeded",
            "answer_text": "parse_rows(path: str) returns list[dict[str, str]].",
            "evidence": [
                {
                    "path": "runtime.py",
                    "line_start": 1,
                    "line_end": 2,
                    "symbol_or_topic": "parse_rows",
                    "excerpt": "def parse_rows(path: str) -> list[dict[str, str]]:",
                    "reason": "Defines the generated source interface.",
                }
            ],
            "limitations": [],
            "trace_summary": ["reading_pm:sufficiency=sufficient"],
        }

    monkeypatch.setattr(code_writing, "run", fake_writing_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)

    response = await propose_code_change({
        "question": "Create source and tests.",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert response["status"] == "succeeded"
    assert len(writing_calls) == 2
    assert len(reading_calls) == 1
    assert "generated_readback:evidence=1" in response["trace_summary"]


async def test_handle_background_coding_task_routes_to_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import supervisor

    reading_calls: list[dict[str, Any]] = []

    async def fake_answer_code_question(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        reading_calls.append(dict(request))
        return {
            "status": "succeeded",
            "answer_text": "Architecture summary.",
            "repository": None,
            "source_scope": None,
            "evidence": [],
            "limitations": [],
            "trace_summary": ["reading:succeeded"],
        }

    async def fail_propose_code_change(_request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("writing interface should not be called")

    monkeypatch.setattr(
        supervisor,
        "_background_coding_router_llm",
        _StaticJsonLLM({
            "operation": "code_reading",
            "reason": "The task asks for a repository explanation.",
        }),
    )
    monkeypatch.setattr(
        supervisor,
        "answer_code_question",
        fake_answer_code_question,
    )
    monkeypatch.setattr(
        supervisor,
        "propose_code_change",
        fail_propose_code_change,
    )

    response = await supervisor.handle_background_coding_task({
        "question": "Summarize the project architecture.",
        "source_summary": "Accepted delayed coding work.",
        "workspace_root": str(tmp_path / "workspace"),
        "max_answer_chars": 1200,
    })

    assert response["status"] == "succeeded"
    assert response["operation"] == "code_reading"
    assert response["answer_text"] == "Architecture summary."
    assert reading_calls == [
        {
            "question": "Summarize the project architecture.",
            "max_answer_chars": 1200,
            "workspace_root": str(tmp_path / "workspace"),
        }
    ]


async def test_handle_background_coding_task_routes_to_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import supervisor

    writing_calls: list[dict[str, Any]] = []

    async def fail_answer_code_question(_request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("reading interface should not be called")

    async def fake_propose_code_change(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        writing_calls.append(dict(request))
        return {
            "status": "succeeded",
            "mode": "create_new_project",
            "answer_text": "Generated proposal.",
            "repository": None,
            "source_scope": None,
            "evidence": [],
            "patch_artifacts": [
                {
                    "artifact_id": "script",
                    "files": ["src/tool.py"],
                    "summary": "Creates source.",
                }
            ],
            "created_files": [{"path": "src/tool.py", "role": "source"}],
            "changed_files": [],
            "validation": {
                "status": "succeeded",
                "parsed": True,
                "sandbox_applied": False,
                "errors": [],
                "warnings": [],
                "files": ["src/tool.py"],
            },
            "external_evidence": [],
            "session": None,
            "limitations": [],
            "trace_summary": ["writing:succeeded"],
        }

    monkeypatch.setattr(
        supervisor,
        "_background_coding_router_llm",
        _StaticJsonLLM({
            "operation": "code_writing",
            "reason": "The task asks for a new code artifact.",
        }),
    )
    monkeypatch.setattr(
        supervisor,
        "answer_code_question",
        fail_answer_code_question,
    )
    monkeypatch.setattr(
        supervisor,
        "propose_code_change",
        fake_propose_code_change,
    )

    response = await supervisor.handle_background_coding_task({
        "question": "Create a Python script.",
        "workspace_root": str(tmp_path / "workspace"),
        "max_answer_chars": 1200,
        "max_artifact_chars": 24000,
    })

    assert response["status"] == "succeeded"
    assert response["operation"] == "code_writing"
    assert response["answer_text"] == "Generated proposal."
    assert response["created_files"] == [
        {"path": "src/tool.py", "role": "source"}
    ]
    assert writing_calls == [
        {
            "question": "Create a Python script.",
            "workspace_root": str(tmp_path / "workspace"),
            "max_answer_chars": 1200,
            "max_artifact_chars": 24000,
        }
    ]


async def test_handle_background_coding_task_fails_on_malformed_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent import supervisor

    async def fail_answer_code_question(_request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("reading interface should not be called")

    async def fail_propose_code_change(_request: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("writing interface should not be called")

    monkeypatch.setattr(
        supervisor,
        "_background_coding_router_llm",
        _StaticJsonLLM({"operation": "invalid"}),
    )
    monkeypatch.setattr(
        supervisor,
        "answer_code_question",
        fail_answer_code_question,
    )
    monkeypatch.setattr(
        supervisor,
        "propose_code_change",
        fail_propose_code_change,
    )

    response = await supervisor.handle_background_coding_task({
        "question": "Summarize this repository.",
    })

    assert response["status"] == "failed"
    assert response["operation"] == "unsupported"
    assert response["answer_text"] == ""
    assert response["limitations"] == [
        "Coding-agent background supervisor returned an invalid operation."
    ]


class _StaticJsonLLM:
    """Return one fixed JSON payload from an LLM-shaped test double."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    async def ainvoke(
        self,
        _messages: list[object],
        *,
        config,
    ) -> SimpleNamespace:
        del config
        content = json.dumps(self._payload, ensure_ascii=False)
        response = SimpleNamespace(content=content)
        return response


def test_coding_agent_readme_documents_reading_and_writing_icd() -> None:
    readme = Path("src/kazusa_ai_chatbot/coding_agent/README.md").read_text(
        encoding="utf-8",
    )

    assert "answer_code_question" in readme
    assert "handle_background_coding_task" in readme
    assert "code_reading" in readme
    assert "BackgroundWorkResult" in readme
    assert "Worker name: `coding_agent`" in readme
    assert "local_root" in readme
    assert "workspace_root" in readme
    assert "cache_key" in readme


def test_code_reading_readme_documents_subagent_icd() -> None:
    readme = Path(
        "src/kazusa_ai_chatbot/coding_agent/code_reading/README.md"
    ).read_text(encoding="utf-8")

    assert "CodeReadingRequest" in readme
    assert "CodeReadingResult" in readme
    assert "Source-scope rules" in readme
    assert "bounded repo-relative evidence" in readme
    assert "`.env` files" in readme
