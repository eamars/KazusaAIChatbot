"""Live RAG route-test harness for retained evidence owners."""

from __future__ import annotations

import pytest


async def run_initializer_case(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    query: str,
    expected_prefixes: list[str],
    required_slot_fragments: list[str] | None = None,
    forbidden_prefixes: list[str] | None = None,
    forbidden_slot_fragments: list[str] | None = None,
) -> list[str]:
    """Gate retired initializer cases until a retained owner is named."""

    del (
        monkeypatch,
        case_id,
        query,
        expected_prefixes,
        required_slot_fragments,
        forbidden_prefixes,
        forbidden_slot_fragments,
    )
    pytest.skip("retired initializer cases have no surviving production owner")
