"""Conditional live diagnostic for direct coding-agent answers."""

import os
from pathlib import Path

import pytest


@pytest.mark.live_llm
async def test_image_reading_question_live_direct_answer() -> None:
    """Run the target direct answer path when live diagnostic config exists."""

    from kazusa_ai_chatbot.coding_agent import answer_code_question

    source_url = os.environ.get("KAZUSA_CODING_AGENT_LIVE_SOURCE_URL")
    workspace_root = os.environ.get("KAZUSA_CODING_AGENT_LIVE_WORKSPACE_ROOT")
    if not source_url or not workspace_root:
        pytest.skip(
            "KAZUSA_CODING_AGENT_LIVE_SOURCE_URL and "
            "KAZUSA_CODING_AGENT_LIVE_WORKSPACE_ROOT are required."
        )

    response = await answer_code_question(
        {
            "question": (
                "[eamars/KazusaAIChatbot]"
                "(https://github.com/eamars/KazusaAIChatbot) "
                '项目是怎么实现读图的'
            ),
            "source_url": source_url,
            "workspace_root": str(Path(workspace_root)),
            "preferred_language": "Chinese",
            "max_answer_chars": 2400,
        }
    )

    assert response["status"] == "succeeded"
    assert response["answer_text"]
    assert response["evidence"]
    assert "base64_data" in response["answer_text"]
    assert "image_observation" in response["answer_text"]
