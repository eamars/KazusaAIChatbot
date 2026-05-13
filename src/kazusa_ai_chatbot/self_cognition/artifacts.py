"""Local artifact writer for self-cognition dry runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.self_cognition import models


def write_tracking_artifacts(
    output_dir: str | Path,
    artifacts: dict[str, Any],
) -> dict[str, str]:
    """Write local self-cognition artifacts under a single directory.

    Args:
        output_dir: Directory owned by the dry-run caller.
        artifacts: Mapping from artifact filename constants to JSON-like data
            or Markdown text.

    Returns:
        Mapping from artifact names to written absolute paths.

    Raises:
        ValueError: If an artifact name is unsupported or path-like.
    """

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, str] = {}
    for artifact_name, payload in artifacts.items():
        if artifact_name not in models.TRACKING_ARTIFACT_NAMES:
            raise ValueError(f"unsupported self-cognition artifact: {artifact_name}")
        if Path(artifact_name).name != artifact_name:
            raise ValueError(f"artifact name must be a filename: {artifact_name}")

        artifact_path = root / artifact_name
        if isinstance(payload, str):
            artifact_path.write_text(payload, encoding="utf-8")
        else:
            rendered = json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            artifact_path.write_text(f"{rendered}\n", encoding="utf-8")
        written_paths[artifact_name] = str(artifact_path)

    return written_paths
