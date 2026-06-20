from pathlib import Path
from typing import Any

import pytest


def _make_repository(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "fixture_repo"
    (repo_root / "src" / "app" / "handlers").mkdir(parents=True)
    (repo_root / "docs").mkdir()
    (repo_root / "tests").mkdir()
    (repo_root / "assets").mkdir()

    (repo_root / "README.md").write_text(
        "\n".join(
            [
                "# Fixture Service",
                "",
                "Run with `fixture-service --host 127.0.0.1 --port 8080`.",
                "The public `/chat` route accepts `ChatRequest` with",
                "`text`, `attachments`, and `metadata` fields.",
                "Images are processed by `ImagePipeline.describe` before",
                "the response planner sees `image_observation`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "architecture.md").write_text(
        "\n".join(
            [
                "# Architecture",
                "",
                "Adapters normalize input and service.py owns orchestration.",
                "background_work.py owns background work routing only.",
                "The checked-in docs mention `/chat` and `fixture-service`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "service.py").write_text(
        "\n".join(
            [
                "from app.background_work import route_background_work",
                "from app.image_pipeline import ImagePipeline",
                "from app.routes import ChatRequest",
                "",
                "",
                "def handle_chat(request: ChatRequest) -> dict:",
                "    pipeline = ImagePipeline()",
                "    observations = [",
                "        pipeline.describe(item)",
                "        for item in request.attachments",
                "        if item.content_type.startswith('image/')",
                "    ]",
                "    route_background_work(request.metadata)",
                "    return {'image_observation': observations}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "routes.py").write_text(
        "\n".join(
            [
                "from dataclasses import dataclass, field",
                "from fastapi import FastAPI",
                "",
                "app = FastAPI()",
                "",
                "",
                "@dataclass",
                "class Attachment:",
                "    content_type: str",
                "    base64_data: str",
                "",
                "",
                "@dataclass",
                "class ChatRequest:",
                "    text: str",
                "    attachments: list[Attachment] = field(default_factory=list)",
                "    metadata: dict = field(default_factory=dict)",
                "",
                "",
                "@app.post('/chat')",
                "def chat_endpoint(request: ChatRequest) -> dict:",
                "    from app.service import handle_chat",
                "",
                "    return handle_chat(request)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "image_pipeline.py").write_text(
        "\n".join(
            [
                "VISION_DESCRIPTOR_LLM = 'vision-descriptor'",
                "",
                "",
                "class ImagePipeline:",
                "    def describe(self, attachment: object) -> dict:",
                "        observation = {",
                "            'model': VISION_DESCRIPTOR_LLM,",
                "            'description': 'short image description',",
                "            'image_observation': {'objects': ['diagram']},",
                "        }",
                "        persist_attachment_description(attachment, observation)",
                "        return observation",
                "",
                "",
                "def persist_attachment_description(",
                "    attachment: object,",
                "    observation: dict,",
                ") -> None:",
                "    attachment.description = observation['description']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "background_work.py").write_text(
        "\n".join(
            [
                "def route_background_work(metadata: dict) -> str:",
                "    if metadata.get('needs_index_refresh'):",
                "        return 'index_refresh'",
                "    return 'none'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "cache.py").write_text(
        "\n".join(
            [
                "IMAGE_CACHE_TTL_SECONDS = 300",
                "",
                "",
                "def cache_failure(cache: dict, cache_key: str, exc: RuntimeError) -> None:",
                "    cache[cache_key] = {'status': 'failed', 'error': str(exc)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "config.py").write_text(
        "\n".join(
            [
                "IMAGE_CACHE_TTL_SECONDS = 300",
                "OPENAI_MODEL = 'fixture-model'",
                "MONGO_COLLECTION = 'images'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "state.py").write_text(
        "\n".join(
            [
                "class ImageState:",
                "    def __init__(self) -> None:",
                "        self.image_observation = None",
                "",
                "    def set_observation(self, observation: dict) -> None:",
                "        self.image_observation = observation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "integrations.py").write_text(
        "\n".join(
            [
                "from pymongo import MongoClient",
                "from openai import OpenAI",
                "",
                "",
                "def save_image_observation(uri: str, observation: dict) -> None:",
                "    MongoClient(uri).fixture.images.insert_one(observation)",
                "",
                "",
                "def call_openai(api_key: str, prompt: str) -> str:",
                "    client = OpenAI(api_key=api_key)",
                "    response = client.responses.create(",
                "        model='fixture-model',",
                "        input=prompt,",
                "    )",
                "    return response.output_text",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "deploy.py").write_text(
        "CLI_COMMAND = 'fixture-service --host 127.0.0.1 --port 8080'\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "handlers" / "debug.py").write_text(
        "\n".join(
            [
                "class Runner:",
                "    def run(self, payload: dict) -> str:",
                "        return 'debug'",
                "",
                "",
                "class DebugHandler:",
                "    def handle(self, payload: dict) -> str:",
                "        return Runner().run(payload)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "app" / "handlers" / "discord.py").write_text(
        "\n".join(
            [
                "class Runner:",
                "    def run(self, payload: dict) -> str:",
                "        return 'discord'",
                "",
                "",
                "class DiscordHandler:",
                "    def handle(self, payload: dict) -> str:",
                "        return Runner().run(payload)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_image_pipeline.py").write_text(
        "\n".join(
            [
                "from app.image_pipeline import ImagePipeline",
                "",
                "",
                "def test_image_pipeline_sets_observation(fake_attachment):",
                "    result = ImagePipeline().describe(fake_attachment)",
                "    assert result['image_observation']['objects'] == ['diagram']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "test_routes.py").write_text(
        "def test_chat_route_exists():\n    assert '/chat'\n",
        encoding="utf-8",
    )
    (repo_root / ".env").write_text(
        "SECRET_TOKEN=do-not-read\n",
        encoding="utf-8",
    )
    (repo_root / "assets" / "logo.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00fixture"
    )

    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "reader",
        "source_url": "https://github.com/fixture/reader",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "a" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "github-fixture-reader-main",
        "dirty_state": "clean",
    }
    return repository


def _repository_scope() -> dict[str, Any]:
    scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://github/fixture/reader",
        "requested_ref": None,
        "interpretation": "entire repository",
    }
    return scope


def _directory_scope() -> dict[str, Any]:
    scope = {
        "kind": "directory",
        "repo_relative_path": "src/app/handlers",
        "source_url": "local://github/fixture/reader/src/app/handlers",
        "requested_ref": None,
        "interpretation": "handlers package",
    }
    return scope


def _file_scope() -> dict[str, Any]:
    scope = {
        "kind": "file",
        "repo_relative_path": "src/app/image_pipeline.py",
        "source_url": (
            "local://github/fixture/reader/src/app/image_pipeline.py"
        ),
        "requested_ref": None,
        "interpretation": "image pipeline file",
    }
    return scope


def _run_reading(
    tmp_path: Path,
    question: str,
    source_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    repository = _make_repository(tmp_path)
    request = {
        "question": question,
        "repository": repository,
        "source_scope": source_scope or _repository_scope(),
        "preferred_language": "English",
        "max_answer_chars": 2400,
    }
    result = run(request)
    return result


def _assert_public_result_shape(result: dict[str, Any]) -> None:
    assert result["status"] in {
        "succeeded",
        "failed",
        "needs_user_input",
        "rejected",
    }
    assert isinstance(result["answer_text"], str)
    assert isinstance(result["evidence"], list)
    assert isinstance(result["limitations"], list)
    assert isinstance(result["trace_summary"], list)
    assert "fixture_repo" not in repr(result)
    assert "workspace" not in repr(result)
    assert "github-fixture-reader-main" not in repr(result)
    assert "SECRET_TOKEN" not in repr(result)


def _assert_evidence_paths(
    result: dict[str, Any],
    expected_paths: set[str],
) -> None:
    _assert_public_result_shape(result)
    assert result["status"] == "succeeded"
    assert result["answer_text"]
    assert result["evidence"]

    actual_paths = {row["path"] for row in result["evidence"]}
    assert actual_paths & expected_paths
    for row in result["evidence"]:
        path = Path(row["path"])
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert not row["path"].startswith(".git")
        assert row["path"] != ".env"
        assert isinstance(row["symbol_or_topic"], str)
        assert isinstance(row["excerpt"], str)
        assert isinstance(row["reason"], str)


def test_answers_feature_or_pipeline_explanation_from_multiple_files(
    tmp_path: Path,
) -> None:
    result = _run_reading(tmp_path, "How does this project read images?")

    _assert_evidence_paths(
        result,
        {
            "src/app/image_pipeline.py",
            "src/app/service.py",
            "src/app/routes.py",
        },
    )
    answer = result["answer_text"]
    assert "ImagePipeline" in answer
    assert "image_observation" in answer


def test_answers_architecture_or_module_responsibility(tmp_path: Path) -> None:
    result = _run_reading(tmp_path, "What owns background work routing?")

    _assert_evidence_paths(
        result,
        {"src/app/background_work.py", "docs/architecture.md"},
    )
    assert "route_background_work" in result["answer_text"]


def test_answers_public_api_route_cli_or_contract_lookup(
    tmp_path: Path,
) -> None:
    result = _run_reading(
        tmp_path,
        "What is the request shape for the /chat endpoint?",
    )

    _assert_evidence_paths(
        result,
        {"src/app/routes.py", "README.md"},
    )
    answer = result["answer_text"]
    assert "/chat" in answer
    assert "ChatRequest" in answer
    assert "attachments" in answer


def test_answers_symbol_class_or_function_explanation(tmp_path: Path) -> None:
    result = _run_reading(tmp_path, "What does ImagePipeline.describe do?")

    _assert_evidence_paths(result, {"src/app/image_pipeline.py"})
    answer = result["answer_text"]
    assert "ImagePipeline" in answer
    assert "describe" in answer


def test_answers_definition_and_usage_search(tmp_path: Path) -> None:
    result = _run_reading(
        tmp_path,
        "Where is VISION_DESCRIPTOR_LLM used?",
    )

    _assert_evidence_paths(result, {"src/app/image_pipeline.py"})
    assert "VISION_DESCRIPTOR_LLM" in result["answer_text"]


def test_answers_file_package_or_directory_summary(tmp_path: Path) -> None:
    result = _run_reading(
        tmp_path,
        "Summarize this folder.",
        source_scope=_directory_scope(),
    )

    _assert_evidence_paths(
        result,
        {
            "src/app/handlers/debug.py",
            "src/app/handlers/discord.py",
        },
    )
    assert "DebugHandler" in result["answer_text"]
    assert "DiscordHandler" in result["answer_text"]


def test_answers_data_config_prompt_or_state_model_reading(
    tmp_path: Path,
) -> None:
    result = _run_reading(
        tmp_path,
        "How is image_observation created and consumed?",
    )

    _assert_evidence_paths(
        result,
        {
            "src/app/image_pipeline.py",
            "src/app/service.py",
            "src/app/state.py",
        },
    )
    assert "image_observation" in result["answer_text"]


def test_answers_error_lifecycle_cache_or_persistence_path(
    tmp_path: Path,
) -> None:
    result = _run_reading(
        tmp_path,
        "How are image failures cached or stored?",
    )

    _assert_evidence_paths(
        result,
        {"src/app/cache.py", "src/app/integrations.py"},
    )
    answer = result["answer_text"]
    assert "cache_failure" in answer
    assert "MongoClient" in answer


def test_answers_test_coverage_mapping(tmp_path: Path) -> None:
    result = _run_reading(
        tmp_path,
        "What tests cover image handling?",
    )

    _assert_evidence_paths(result, {"tests/test_image_pipeline.py"})
    assert "test_image_pipeline_sets_observation" in result["answer_text"]


def test_answers_dependency_or_external_integration_usage(
    tmp_path: Path,
) -> None:
    result = _run_reading(
        tmp_path,
        "How does it call Mongo, OpenAI, or FastAPI?",
    )

    _assert_evidence_paths(
        result,
        {"src/app/integrations.py", "src/app/routes.py"},
    )
    answer = result["answer_text"]
    assert "MongoClient" in answer
    assert "OpenAI" in answer
    assert "FastAPI" in answer


def test_answers_intra_repo_comparison(tmp_path: Path) -> None:
    result = _run_reading(
        tmp_path,
        "Compare DebugHandler and DiscordHandler.",
    )

    _assert_evidence_paths(
        result,
        {
            "src/app/handlers/debug.py",
            "src/app/handlers/discord.py",
        },
    )
    assert "DebugHandler" in result["answer_text"]
    assert "DiscordHandler" in result["answer_text"]


def test_answers_docs_to_code_consistency(tmp_path: Path) -> None:
    result = _run_reading(
        tmp_path,
        "Does README match the /chat implementation?",
    )

    _assert_evidence_paths(result, {"README.md", "src/app/routes.py"})
    assert "/chat" in result["answer_text"]
    assert "ChatRequest" in result["answer_text"]


def test_answers_static_impact_or_risk_read(tmp_path: Path) -> None:
    result = _run_reading(
        tmp_path,
        "What might depend on ImagePipeline?",
    )

    _assert_evidence_paths(
        result,
        {"src/app/image_pipeline.py", "src/app/service.py"},
    )
    assert "ImagePipeline" in result["answer_text"]
    assert "handle_chat" in result["answer_text"]


def test_answers_build_run_or_deployment_reading(tmp_path: Path) -> None:
    result = _run_reading(tmp_path, "How do I run this project?")

    _assert_evidence_paths(result, {"README.md", "src/app/deploy.py"})
    assert "fixture-service" in result["answer_text"]


def test_broad_question_succeeds_with_limitations_or_asks_for_scope(
    tmp_path: Path,
) -> None:
    result = _run_reading(tmp_path, "Explain everything in this repository.")

    _assert_public_result_shape(result)
    assert result["status"] in {"succeeded", "needs_user_input"}
    if result["status"] == "succeeded":
        assert result["limitations"]
        assert result["evidence"]
    else:
        assert result["answer_text"]
        assert not result["evidence"]


def test_ambiguous_symbol_returns_needs_user_input(tmp_path: Path) -> None:
    result = _run_reading(tmp_path, "What does Runner do?")

    _assert_public_result_shape(result)
    assert result["status"] == "needs_user_input"
    assert result["answer_text"]
    assert not result["evidence"]


def test_incomplete_evidence_reports_limitations_instead_of_hallucinating(
    tmp_path: Path,
) -> None:
    result = _run_reading(
        tmp_path,
        "How is webhook replay persisted after image failure?",
    )

    _assert_public_result_shape(result)
    assert result["status"] == "succeeded"
    assert result["limitations"]
    assert "ReplayStore" not in result["answer_text"]
    assert "webhook_replay" not in result["answer_text"]


def test_file_scope_keeps_evidence_within_file_unless_trace_names_lookup(
    tmp_path: Path,
) -> None:
    result = _run_reading(
        tmp_path,
        "Summarize this file and mention visible callers.",
        source_scope=_file_scope(),
    )

    _assert_public_result_shape(result)
    assert result["status"] == "succeeded"
    outside_scope_paths = [
        row["path"]
        for row in result["evidence"]
        if row["path"] != "src/app/image_pipeline.py"
    ]
    if outside_scope_paths:
        trace_text = " ".join(result["trace_summary"]).lower()
        assert "repo-wide" in trace_text
        assert "symbol lookup" in trace_text


@pytest.mark.parametrize(
    ("repo_relative_path", "expected_reason"),
    [
        ("../outside.py", "outside the repository"),
        (".git/config", ".git internals"),
        (".env", "environment files"),
        ("src/app/secret_token.py", "secret-like files"),
        ("assets/logo.png", "binary assets"),
    ],
)
def test_rejects_unsafe_source_scopes_before_reading(
    tmp_path: Path,
    repo_relative_path: str,
    expected_reason: str,
) -> None:
    source_scope = {
        "kind": "file",
        "repo_relative_path": repo_relative_path,
        "source_url": f"local://github/fixture/reader/{repo_relative_path}",
        "requested_ref": None,
        "interpretation": "unsafe file scope",
    }

    result = _run_reading(
        tmp_path / repo_relative_path.replace("/", "_").replace(".", "dot"),
        "Summarize this file.",
        source_scope=source_scope,
    )

    _assert_public_result_shape(result)
    assert result["status"] == "rejected"
    assert expected_reason in result["answer_text"]
    assert result["evidence"] == []


def test_answer_respects_public_answer_cap(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    result = run(
        {
            "question": "How does this project read images?",
            "repository": _make_repository(tmp_path),
            "source_scope": _repository_scope(),
            "preferred_language": "English",
            "max_answer_chars": 160,
        }
    )

    _assert_public_result_shape(result)
    assert result["status"] == "succeeded"
    assert len(result["answer_text"]) <= 160


def test_rejects_explicitly_unsupported_reading_requests(
    tmp_path: Path,
) -> None:
    unsupported_questions = [
        "Apply a patch that rewrites ImagePipeline.describe.",
        "Run pytest and tell me whether it passes.",
        "Install pillow before answering this question.",
        "Inspect .env and tell me SECRET_TOKEN.",
        "Dump the full raw contents of src/app/image_pipeline.py.",
        "Analyze the binary pixels in assets/logo.png.",
        "Use git@github.com:fixture/private.git to inspect the private repo.",
        "Certify that this repository is legally compliant and secure.",
        "What is the latest FastAPI CVE status today?",
    ]

    for index, question in enumerate(unsupported_questions):
        result = _run_reading(tmp_path / f"case_{index}", question)

        _assert_public_result_shape(result)
        assert result["status"] == "rejected", question
        assert result["limitations"], question
        assert result["evidence"] == []
