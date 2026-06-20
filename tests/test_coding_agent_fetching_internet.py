from pathlib import Path

import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching import run


PUBLIC_CODE_SOURCES = [
    "https://github.com/octocat/Hello-World",
    "https://github.com/octocat/Spoon-Knife",
    "https://github.com/github/gitignore",
    "https://github.com/github/gitignore/tree/main/Global",
    "https://github.com/github/gitignore/blob/main/Python.gitignore",
    "https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore",
    "https://github.com/pypa/sampleproject",
    "https://github.com/pallets/itsdangerous",
    "https://github.com/pallets/markupsafe",
    "https://github.com/pallets/click",
]


@pytest.mark.live_internet
async def test_fetching_resolves_ten_public_code_sources(tmp_path: Path) -> None:
    workspace_root = tmp_path / "coding_workspace"
    resolved_sources: list[str] = []

    for source_url in PUBLIC_CODE_SOURCES:
        result = await run(
            {
                "source_url": source_url,
                "workspace_root": str(workspace_root),
            }
        )

        assert result["status"] == "succeeded", result
        assert result["repository"] is not None
        assert result["source_scope"] is not None
        resolved_sources.append(source_url)

    assert resolved_sources == PUBLIC_CODE_SOURCES
