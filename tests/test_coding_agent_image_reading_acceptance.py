from pathlib import Path
from typing import Any

import pytest


def _make_image_repository(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "image_repo"
    (repo_root / "src" / "adapters").mkdir(parents=True)
    (repo_root / "src" / "brain").mkdir(parents=True)
    (repo_root / "src" / "nodes").mkdir(parents=True)
    (repo_root / "src" / "history").mkdir(parents=True)

    (repo_root / "src" / "adapters" / "debug_adapter.py").write_text(
        "\n".join(
            [
                "def normalize_attachment(upload: dict) -> dict:",
                "    return {",
                "        'content_type': upload['content_type'],",
                "        'base64_data': upload['base64_data'],",
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "brain" / "service.py").write_text(
        "\n".join(
            [
                "from nodes.media_descriptor import multimedia_descriptor_agent",
                "",
                "",
                "def build_user_multimedia_input(message: dict) -> list[dict]:",
                "    return message.get('attachments', [])",
                "",
                "",
                "def run_turn(message: dict) -> dict:",
                "    user_multimedia_input = build_user_multimedia_input(message)",
                "    image_observation = multimedia_descriptor_agent(",
                "        user_multimedia_input",
                "    )",
                "    return {'image_observation': image_observation}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "nodes" / "media_descriptor.py").write_text(
        "\n".join(
            [
                "VISION_DESCRIPTOR_LLM = 'vision-descriptor'",
                "",
                "",
                "def multimedia_descriptor_agent(items: list[dict]) -> list[dict]:",
                "    observations = []",
                "    for item in items:",
                "        observations.append(",
                "            {",
                "                'description': 'image description',",
                "                'image_observation': {'source': VISION_DESCRIPTOR_LLM},",
                "            }",
                "        )",
                "    return observations",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "nodes" / "cognition_episode.py").write_text(
        "\n".join(
            [
                "def project_media_percept(image_observation: dict) -> dict:",
                "    return {'typed_media_percept': image_observation}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "history" / "attachments.py").write_text(
        "\n".join(
            [
                "def project_history_attachment(description: str) -> str:",
                "    return f'<image>{description}</image>'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text(
        "\n".join(
            [
                "# Image Fixture",
                "",
                "Adapters send base64_data. The service builds",
                "user_multimedia_input and calls multimedia_descriptor_agent.",
                "Descriptions are projected as image_observation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repository = {
        "provider": "github",
        "owner": "eamars",
        "repo": "KazusaAIChatbot",
        "source_url": "https://github.com/eamars/KazusaAIChatbot",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "c" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "github-eamars-KazusaAIChatbot-main",
        "dirty_state": "clean",
    }
    return repository


def _repository_scope() -> dict[str, Any]:
    scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://github/eamars/KazusaAIChatbot",
        "requested_ref": None,
        "interpretation": "entire repository",
    }
    return scope


def test_target_image_reading_question_returns_evidence_backed_answer(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    result = run(
        {
            "question": '[eamars/KazusaAIChatbot](https://github.com/eamars/KazusaAIChatbot) 项目是怎么实现读图的',
            "repository": _make_image_repository(tmp_path),
            "source_scope": _repository_scope(),
            "preferred_language": "Chinese",
            "max_answer_chars": 2400,
        }
    )

    assert result["status"] == "succeeded"
    assert result["answer_text"]
    evidence_paths = {row["path"] for row in result["evidence"]}
    assert "src/adapters/debug_adapter.py" in evidence_paths
    assert "src/brain/service.py" in evidence_paths
    assert "src/nodes/media_descriptor.py" in evidence_paths
    assert "src/nodes/cognition_episode.py" in evidence_paths
    assert "src/history/attachments.py" in evidence_paths
    answer = result["answer_text"]
    assert "base64_data" in answer
    assert "user_multimedia_input" in answer
    assert "multimedia_descriptor_agent" in answer
    assert "VISION_DESCRIPTOR_LLM" in answer
    assert "image_observation" in answer
    assert "<image>" in answer
    assert "local_root" not in repr(result)
    assert "workspace_root" not in repr(result)
    assert "cache_key" not in repr(result)


@pytest.mark.asyncio
async def test_target_image_question_uses_phase0_embedded_url_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question
    from kazusa_ai_chatbot.coding_agent.code_fetching import managed_clone

    repository = _make_image_repository(tmp_path)

    def fake_ensure_managed_checkout(source, workspace_root: str):
        assert source.owner == "eamars"
        assert source.repo == "KazusaAIChatbot"
        assert workspace_root == str(tmp_path / "workspace")
        return repository

    monkeypatch.setattr(
        managed_clone,
        "ensure_managed_checkout",
        fake_ensure_managed_checkout,
    )

    response = await answer_code_question(
        {
            "question": (
                "[eamars/KazusaAIChatbot]"
                "(https://github.com/eamars/KazusaAIChatbot) "
                '项目是怎么实现读图的'
            ),
            "workspace_root": str(tmp_path / "workspace"),
            "preferred_language": "Chinese",
            "max_answer_chars": 2400,
        }
    )

    assert response["status"] == "succeeded"
    assert response["repository"] is not None
    assert response["repository"]["owner"] == "eamars"
    assert response["repository"]["repo"] == "KazusaAIChatbot"
    assert response["source_scope"] is not None
    assert response["source_scope"]["kind"] == "repository"
    assert "base64_data" in response["answer_text"]
    assert "user_multimedia_input" in response["answer_text"]
    assert "multimedia_descriptor_agent" in response["answer_text"]
    assert "VISION_DESCRIPTOR_LLM" in response["answer_text"]
    assert "image_observation" in response["answer_text"]
    assert "<image>" in response["answer_text"]
    assert "local_root" not in repr(response)
    assert "workspace_root" not in repr(response)
    assert "cache_key" not in repr(response)
