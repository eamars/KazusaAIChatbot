"""Shared attachment storage policy for message envelopes.

This module owns public constants used by adapter attachment handlers and the
conversation storage path. Inputs are normalized attachment references; outputs
are policy decisions made by callers using these shared limits. New attachment
modalities can reuse the constants without importing concrete handlers.
"""

from __future__ import annotations

INLINE_ATTACHMENT_BYTE_LIMIT = 256 * 1024
