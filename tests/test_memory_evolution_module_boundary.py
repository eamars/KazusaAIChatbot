"""Static checks for memory-evolution module boundaries."""

from __future__ import annotations

from pathlib import Path


def test_memory_evolution_does_not_touch_mongodb_directly() -> None:
    """Memory evolution must use the DB interface instead of Motor handles."""
    root = Path("src/kazusa_ai_chatbot/memory_evolution")
    forbidden = (
        "get_db",
        "db.memory",
        ".insert_one(",
        ".update_one(",
        ".update_many(",
        ".delete_one(",
        ".delete_many(",
        ".replace_one(",
        ".count_documents(",
        ".$vectorSearch",
    )

    offenders: list[str] = []
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{path}:{token}")

    assert offenders == []
