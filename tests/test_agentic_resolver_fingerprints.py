"""Deterministic authority and semantic-operation fingerprint tests."""

from __future__ import annotations

from agentic_resolver.fingerprints import (
    audience_fingerprint,
    operation_payload_digest,
    policy_fingerprint,
    profile_fingerprint,
    scope_fingerprint,
    tool_catalog_digest,
)


def test_authority_fingerprints_are_stable_and_do_not_hash_event_logs() -> None:
    """Authority identities remain stable for equivalent mappings."""

    scope = {"workspace": "standalone", "permissions": ["read"]}
    audience = {"kind": "operator", "ids": ["u_1"]}
    profile = {"profile": "kazusa-resolver-v1", "release": "0.1.1-rc.2"}
    catalog = [{"name": "submit_resolution", "version": "1"}]
    policy = {"epoch": "2026-08-28.1", "max_steps": 4}
    first = (
        scope_fingerprint(scope),
        audience_fingerprint(audience),
        profile_fingerprint(profile),
        tool_catalog_digest(catalog),
        policy_fingerprint(policy),
    )
    second = (
        scope_fingerprint({"permissions": ["read"], "workspace": "standalone"}),
        audience_fingerprint({"ids": ["u_1"], "kind": "operator"}),
        profile_fingerprint({"release": "0.1.1-rc.2", "profile": "kazusa-resolver-v1"}),
        tool_catalog_digest(catalog),
        policy_fingerprint({"max_steps": 4, "epoch": "2026-08-28.1"}),
    )
    assert first == second
    assert all(value.startswith("sha256:") for value in first)


def test_operation_payload_digest_is_canonical_and_excludes_transport_fields() -> None:
    """JSON-RPC ids, bearer credentials, and semantic ids do not affect digest."""

    left = {
        "id": 1,
        "method": "resolution.open",
        "params": {
            "operation_id": "op_1",
            "operation_payload_digest": "ignored",
            "token": "secret-a",
            "runtime": {"b": 2, "a": 1},
        },
    }
    right = {
        "id": 999,
        "method": "resolution.open",
        "params": {
            "operation_id": "op_other",
            "operation_payload_digest": "ignored-again",
            "token": "secret-b",
            "runtime": {"a": 1, "b": 2},
        },
    }
    assert operation_payload_digest(left) == operation_payload_digest(right)
    assert operation_payload_digest(left).startswith("sha256:")
