"""Process-local image payload cache scoped to one conversation participant."""

from __future__ import annotations

import base64
import binascii
from collections import OrderedDict
from time import monotonic
from uuid import uuid4

from kazusa_ai_chatbot.config import (
    MEDIA_SESSION_CACHE_MAX_BYTES_PER_SCOPE,
    MEDIA_SESSION_CACHE_MAX_ITEM_BYTES,
    MEDIA_SESSION_CACHE_MAX_ITEMS_PER_SCOPE,
    MEDIA_SESSION_CACHE_TTL_SECONDS,
)

SessionMediaScope = tuple[str, str, str]
_SESSION_MEDIA_CACHE: dict[SessionMediaScope, OrderedDict[str, dict[str, object]]] = {}


def put_session_media(
    scope: SessionMediaScope,
    media_items: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Store bounded image payloads and return trusted cache references."""

    _validate_scope(scope)
    bucket = _SESSION_MEDIA_CACHE.setdefault(scope, OrderedDict())
    _evict_expired(bucket)
    refs: list[dict[str, object]] = []
    for item in media_items:
        payload = _validated_payload(item)
        if payload is None:
            continue
        cache_ref = uuid4().hex
        payload["cache_ref"] = cache_ref
        bucket[cache_ref] = payload
        refs.append(_public_ref(payload))
    _enforce_limits(bucket)
    return refs


def begin_session_turn(scope: SessionMediaScope) -> None:
    """Mark retained media from prior turns as recent before current intake."""

    _validate_scope(scope)
    bucket = _SESSION_MEDIA_CACHE.get(scope)
    if bucket is None:
        return
    _evict_expired(bucket)
    for payload in bucket.values():
        payload["turn_relation"] = "recent"


def list_session_media_refs(scope: SessionMediaScope) -> list[dict[str, object]]:
    """List prompt-safe current cache references in stable insertion order."""

    _validate_scope(scope)
    bucket = _SESSION_MEDIA_CACHE.get(scope)
    if bucket is None:
        return []
    _evict_expired(bucket)
    return [_public_ref(payload) for payload in bucket.values()]


def get_session_media(
    scope: SessionMediaScope,
    cache_ref: str,
) -> dict[str, object] | None:
    """Return one trusted payload only from the exact owning scope."""

    _validate_scope(scope)
    if not isinstance(cache_ref, str) or not cache_ref.strip():
        return None
    bucket = _SESSION_MEDIA_CACHE.get(scope)
    if bucket is None:
        return None
    _evict_expired(bucket)
    payload = bucket.get(cache_ref)
    if payload is None:
        return None
    payload["last_seen_monotonic"] = monotonic()
    bucket.move_to_end(cache_ref)
    result = dict(payload)
    return result


def clear_session_media(scope: SessionMediaScope | None = None) -> None:
    """Clear one scope or the whole process-local image cache."""

    if scope is None:
        _SESSION_MEDIA_CACHE.clear()
        return
    _validate_scope(scope)
    _SESSION_MEDIA_CACHE.pop(scope, None)


def _validated_payload(item: dict[str, object]) -> dict[str, object] | None:
    """Return a bounded image payload suitable for trusted process memory."""

    if item.get("media_kind") != "image":
        return None
    content_type = item.get("content_type")
    base64_data = item.get("base64_data")
    if (
        not isinstance(content_type, str)
        or not content_type.lower().startswith("image/")
        or not isinstance(base64_data, str)
        or not base64_data.strip()
    ):
        return None
    try:
        byte_count = len(base64.b64decode(base64_data, validate=True))
    except (binascii.Error, ValueError):
        return None
    if byte_count > MEDIA_SESSION_CACHE_MAX_ITEM_BYTES:
        return None
    source_summary = item.get("source_summary")
    if not isinstance(source_summary, str):
        source_summary = ""
    descriptor = item.get("existing_descriptor")
    if not isinstance(descriptor, str):
        descriptor = ""
    timestamp = monotonic()
    payload = {
        "media_kind": "image",
        "content_type": content_type.strip().lower(),
        "base64_data": base64_data,
        "byte_count": byte_count,
        "source_summary": source_summary.strip(),
        "existing_descriptor": descriptor.strip(),
        "turn_relation": "current",
        "created_at_monotonic": timestamp,
        "last_seen_monotonic": timestamp,
    }
    return payload


def _public_ref(payload: dict[str, object]) -> dict[str, object]:
    """Return trusted lookup metadata without raw image data."""

    result = {
        "cache_ref": payload["cache_ref"],
        "media_kind": "image",
        "content_type": payload["content_type"],
        "source_summary": payload["source_summary"],
        "turn_relation": payload["turn_relation"],
    }
    return result


def _evict_expired(bucket: OrderedDict[str, dict[str, object]]) -> None:
    """Remove entries older than the fixed session TTL."""

    cutoff = monotonic() - MEDIA_SESSION_CACHE_TTL_SECONDS
    expired_refs = [
        cache_ref
        for cache_ref, payload in bucket.items()
        if float(payload["created_at_monotonic"]) < cutoff
    ]
    for cache_ref in expired_refs:
        bucket.pop(cache_ref, None)


def _enforce_limits(bucket: OrderedDict[str, dict[str, object]]) -> None:
    """Evict oldest entries until all fixed cache caps are met."""

    while bucket and (
        len(bucket) > MEDIA_SESSION_CACHE_MAX_ITEMS_PER_SCOPE
        or _bucket_bytes(bucket) > MEDIA_SESSION_CACHE_MAX_BYTES_PER_SCOPE
    ):
        bucket.popitem(last=False)


def _bucket_bytes(bucket: OrderedDict[str, dict[str, object]]) -> int:
    """Return total decoded image bytes in one scoped cache bucket."""

    result = sum(int(payload["byte_count"]) for payload in bucket.values())
    return result


def _validate_scope(scope: SessionMediaScope) -> None:
    """Require the exact platform, channel, and global-user cache scope."""

    if (
        not isinstance(scope, tuple)
        or len(scope) != 3
        or any(not isinstance(value, str) or not value.strip() for value in scope)
    ):
        raise ValueError("scope: expected platform, channel, and global user")
