from __future__ import annotations

from datetime import datetime, timezone

import kazusa_ai_chatbot.logging_retention as retention


def test_expiry_from_iso_uses_supplied_timestamp():
    expires_at = retention.expiry_from_storage_iso(
        "2026-01-01T00:00:00+00:00",
        ttl_days=2,
    )

    assert expires_at == datetime(2026, 1, 3, tzinfo=timezone.utc)


def test_expiry_from_iso_falls_back_for_bad_timestamp(monkeypatch):
    fixed_now = datetime(2026, 1, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(retention, "storage_utc_now", lambda: fixed_now)

    expires_at = retention.expiry_from_storage_iso("not-a-date", ttl_days=1)

    assert expires_at == datetime(2026, 1, 6, tzinfo=timezone.utc)
