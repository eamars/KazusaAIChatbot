"""Canonical deterministic identity helpers for resolver authority."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from agentic_resolver.errors import ResolverContractError

_TRANSPORT_FIELDS = frozenset({
    "id", "jsonrpc", "token", "authorization", "operation_id",
    "operation_payload_digest", "protocol_version",
})


def canonical_json(value: object) -> str:
    """Return the canonical JSON representation used by all digests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ResolverContractError("value is not canonical JSON") from exc


def _digest(value: object) -> str:
    payload = canonical_json(value).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def scope_fingerprint(scope: Mapping[str, Any]) -> str:
    return _digest(scope)


def audience_fingerprint(audience: Mapping[str, Any]) -> str:
    return _digest(audience)


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    return _digest(profile)


def tool_catalog_digest(catalog: Sequence[Mapping[str, Any]]) -> str:
    return _digest(catalog)


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    return _digest(policy)


def operation_payload_digest(frame: Mapping[str, Any]) -> str:
    """Digest a method and immutable params, excluding transport identities."""

    method = frame.get("method")
    params = frame.get("params", {})
    if not isinstance(method, str) or not method:
        raise ResolverContractError("operation method is required")
    if not isinstance(params, Mapping):
        raise ResolverContractError("operation params must be an object")
    immutable = {
        key: value
        for key, value in params.items()
        if key.lower() not in _TRANSPORT_FIELDS
    }
    return _digest({"method": method, "params": immutable})
