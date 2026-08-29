"""Deterministic contract gates for the semantic gateway."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest


def test_public_gateway_exports_are_bounded_contracts_only() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway import (
        KazusaSemanticCapabilityResultV1,
        SEMANTIC_TOOL_NAMES,
    )

    assert len(SEMANTIC_TOOL_NAMES) == 13
    assert all(name.startswith("kazusa_") for name in SEMANTIC_TOOL_NAMES)
    assert not any(name.endswith("_v1") for name in SEMANTIC_TOOL_NAMES)
    assert KazusaSemanticCapabilityResultV1


def test_semantic_result_uses_entities_opaque_page_refs_and_idempotent_mutation_outcomes() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
        KazusaSemanticCapabilityResultV1,
    )

    result = KazusaSemanticCapabilityResultV1.from_mapping({
        "schema_version": "kazusa_semantic_capability_result.v1",
        "status": "ok",
        "entities": [{
            "memory_ref": "memref_opaque",
            "information": "A document about MongoDB is relevant to this memory.",
        }],
        "page": {"has_more": True, "next_page_ref": "page_opaque"},
        "evidence": [{
            "receipt_id": "receipt_1",
            "source_kind": "conversation",
            "semantic_ref": "memref_opaque",
            "content_digest": "sha256:abc",
            "occurred_at": "2026-08-28T00:00:00Z",
        }],
        "mutation": {
            "outcome": "committed",
            "semantic_ref": "memref_opaque",
            "idempotency_key": "idem_1",
        },
        "error": None,
    })

    assert result.to_dict()["page"]["next_page_ref"] == "page_opaque"
    assert result.to_dict()["mutation"]["idempotency_key"] == "idem_1"
    semantic_text = KazusaSemanticCapabilityResultV1.from_mapping({
        **result.to_dict(),
        "entities": [{
            "information": "A document about MongoDB is legitimate semantic content.",
        }],
    })
    assert "MongoDB" in semantic_text.entities[0]["information"]


def test_description_stripped_catalog_is_self_explanatory_storage_independent_and_excludes_standard_capabilities() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.catalog import (
        description_stripped_catalog,
    )

    catalog = description_stripped_catalog({"read_file", "submit_resolution"})
    names = {item["name"] for item in catalog}
    assert len(names) == 13
    assert "read_file" not in names
    assert "submit_resolution" not in names
    assert all("description" not in item for item in catalog)

    schemas = {
        item["name"]: item["input_schema"]
        for item in catalog
    }
    assert set(schemas) == {
        "kazusa_search_conversation_history",
        "kazusa_read_conversation_entries",
        "kazusa_summarize_conversation_participants",
        "kazusa_search_memories",
        "kazusa_read_memories",
        "kazusa_remember_information",
        "kazusa_revise_memory",
        "kazusa_change_memory_lifecycle",
        "kazusa_find_people_by_name",
        "kazusa_read_person_profiles",
        "kazusa_recall_active_context",
        "kazusa_read_calendar_context",
        "kazusa_inspect_attached_media",
    }
    for schema in schemas.values():
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False

    conversation_search = schemas["kazusa_search_conversation_history"]
    assert set(conversation_search["properties"]) == {
        "query", "time_range", "max_results", "next_page_ref",
    }
    assert set(conversation_search["properties"]["time_range"]["properties"]) == {
        "start_at", "end_at",
    }
    assert conversation_search["required"] == ["query"]

    participant_summary = schemas["kazusa_summarize_conversation_participants"]
    assert set(participant_summary["properties"]) == {
        "time_range", "max_people", "next_page_ref",
    }

    memory_search = schemas["kazusa_search_memories"]
    assert set(memory_search["properties"]) == {
        "query", "subject_scope", "memory_kinds", "max_results",
        "next_page_ref",
    }
    assert memory_search["properties"]["subject_scope"]["enum"] == [
        "current_user", "active_character", "shared_world", "all",
    ]
    assert memory_search["properties"]["memory_kinds"]["items"]["enum"] == [
        "profile_fact", "relationship", "commitment", "experience",
        "world_knowledge",
    ]

    remember = schemas["kazusa_remember_information"]
    assert remember["required"] == [
        "subject", "information", "memory_kind", "reason", "provenance",
    ]
    assert "idempotency_key" not in remember["properties"]
    assert remember["properties"]["provenance"]["type"] == "object"
    assert set(remember["properties"]["provenance"]["properties"]) == {
        "conversation_entry_ref", "current_task",
    }
    for name in ("kazusa_revise_memory", "kazusa_change_memory_lifecycle"):
        assert "idempotency_key" not in schemas[name]["properties"]

    people = schemas["kazusa_find_people_by_name"]
    assert set(people["properties"]) == {
        "display_name", "match_relation", "max_results", "next_page_ref",
    }
    assert people["properties"]["match_relation"]["enum"] == [
        "exact", "contains", "starts_with", "ends_with",
    ]

    recall = schemas["kazusa_recall_active_context"]
    assert recall["properties"]["kinds"]["items"]["enum"] == [
        "commitments", "progress", "history", "calendar",
    ]
    calendar = schemas["kazusa_read_calendar_context"]
    assert set(calendar["properties"]) == {"view", "max_results", "next_page_ref"}


def test_opaque_reference_is_sealed_complete_authority_bound_and_restart_resolvable() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import SemanticActivationAuthorityV1
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
        OpaqueReferenceCodec,
        content_digest,
    )

    scope = {
        "platform": "debug",
        "platform_channel_id": "channel",
        "global_user_id": "user",
    }
    workspace = "C:/workspace/project"
    now = datetime.now(UTC)
    authority = SemanticActivationAuthorityV1(
        activation_id="act-ref",
        lease_epoch=1,
        resolution_thread_id="thread-ref",
        segment_id="segment-ref",
        brain_conversation_ref="chat:debug:ref",
        service_scope=scope,
        scope_fingerprint=content_digest(scope),
        audience_fingerprint="sha256:audience",
        workspace_root=workspace,
        route_digest="sha256:route",
        catalog_digest="sha256:catalog",
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest="sha256:route",
        workspace_fingerprint=content_digest({"workspace_root": workspace}),
        issued_reference_digest="sha256:issued",
        policy_epoch="dsh-standard-policy-v2",
        interaction_issuer="dsh-sidecar-test",
        issued_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        token_id="token-ref",
        nonce="nonce-ref",
    )
    value = {"source_id": "known-document-id", "cache_ref": "known-backend-cursor"}
    codec = OpaqueReferenceCodec(b"reference-secret").with_authority(authority)
    reference = codec.issue("memory_ref", value)
    assert "known-document-id" not in reference
    assert "known-backend-cursor" not in reference
    assert "source_id" not in reference
    assert "cache_ref" not in reference
    assert len(reference.encode("utf-8")) < 1024
    for authority_value in (
        "act-ref",
        "thread-ref",
        "segment-ref",
        "chat:debug:ref",
        "C:/workspace/project",
        "sha256:audience",
        "sha256:route",
        "sha256:catalog",
        "kazusa-resolver-standard-v2",
        "dsh-standard-policy-v2",
        "dsh-sidecar-test",
        "token-ref",
        "nonce-ref",
    ):
        assert authority_value not in reference
    with pytest.raises(ValueError):
        codec.issue(
            "memory_ref",
            value,
            authority={"authority_digest": "sha256:caller-supplied"},
        )

    restarted = OpaqueReferenceCodec(b"reference-secret").with_authority(authority)
    assert restarted.resolve(reference, "memory_ref") == value

    other_scope = replace(
        authority,
        service_scope={**scope, "global_user_id": "other-user"},
        scope_fingerprint=content_digest({
            **scope,
            "global_user_id": "other-user",
        }),
    )
    with pytest.raises(ValueError):
        restarted.resolve(reference, "memory_ref", authority=other_scope)
    with pytest.raises(ValueError):
        restarted.resolve(
            reference,
            "memory_ref",
            authority=replace(authority, segment_id="other-segment"),
        )
    with pytest.raises(ValueError):
        restarted.resolve(
            reference,
            "memory_ref",
            authority=replace(
                authority,
                expires_at=(now + timedelta(minutes=4)).isoformat().replace(
                    "+00:00",
                    "Z",
                ),
            ),
        )
