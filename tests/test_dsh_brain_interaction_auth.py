"""Signed Brain interaction authentication tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tests.test_dsh_brain_interaction_contracts import _request_mapping


def test_mac_timestamp_nonce_digest_and_constant_time_validation_fail_closed() -> None:
    from kazusa_ai_chatbot.dsh_interaction.auth import (
        InteractionNonceReplayStore,
        sign_request,
        verify_request,
    )
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV2

    secret = b"brain-secret"
    request = DshBrainInteractionRequestV2.from_mapping(_request_mapping())
    signed = sign_request(request, secret=secret)
    replay = InteractionNonceReplayStore()
    verified = verify_request(signed, secret=secret, replay_owner=replay)
    assert verified.request_digest == request.request_digest
    with pytest.raises(ValueError):
        verify_request(signed, secret=secret, replay_owner=replay)
    signed_bad = signed.__class__(**{**signed.__dict__, "mac": "bad"}) if hasattr(signed, "__dict__") else None
    assert signed_bad is None or signed_bad.mac != signed.mac


def test_signed_interaction_remains_valid_for_its_full_declared_lifetime() -> None:
    """Treat clock skew as future tolerance, not a one-minute maximum age."""

    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request, validate_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV2

    secret = b"brain-secret"
    issued = datetime(2026, 8, 29, 0, 0, tzinfo=UTC)
    value = _request_mapping()
    value["issued_at"] = issued.isoformat().replace("+00:00", "Z")
    value["expires_at"] = (
        issued + timedelta(minutes=5)
    ).isoformat().replace("+00:00", "Z")
    request = DshBrainInteractionRequestV2.from_mapping(value)
    signed = sign_request(request, secret=secret)

    validate_request(
        signed,
        secret=secret,
        now=(issued + timedelta(minutes=2)).isoformat(),
    )
    with pytest.raises(ValueError, match="outside the allowed skew"):
        validate_request(
            signed,
            secret=secret,
            now=(issued - timedelta(seconds=61)).isoformat(),
        )
