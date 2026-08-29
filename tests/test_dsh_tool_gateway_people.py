"""People semantic service tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_people_services_return_semantic_candidates_profiles_and_opaque_person_refs() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.people import PeopleSemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec

    async def find(name, **kwargs):
        assert name == "Alice"
        return [{"global_user_id": "user-1", "display_name": "Alice", "platform": "debug"}]

    async def read(global_user_id):
        assert global_user_id == "user-1"
        return {"platform_accounts": [{"display_name": "Alice"}], "facts": ["likes tea"]}

    service = PeopleSemanticService(
        codec=OpaqueReferenceCodec(b"people-test-secret"),
        find=find,
        read=read,
    )
    found = await service.find_people_by_name(
        display_name="Alice",
        match_relation="exact",
    )
    reference = found.entities[0]["person_ref"]
    profile = await service.read_person_profiles(person_refs=[reference])
    assert profile.entities[0]["known_names"] == ["Alice"]
    assert profile.evidence[0].semantic_ref == reference
