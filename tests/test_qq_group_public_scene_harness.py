"""Deterministic contract checks for the guarded group-scene live harness."""

from __future__ import annotations

import pytest

from tests.test_qq_group_public_scene_live_llm import (
    _assert_visible_hard_boundaries,
)


def test_visible_hard_boundaries_accept_nonempty_surface() -> None:
    """The deterministic harness accepts a nonempty response without leaks."""

    result = {
        'turns': [{
            'response': ['A bounded visible response.'],
        }],
    }

    _assert_visible_hard_boundaries(result)


def test_visible_hard_boundaries_reject_empty_surface() -> None:
    """The deterministic harness requires a visible response when expected."""

    result = {
        'turns': [{
            'response': [],
        }],
    }

    with pytest.raises(AssertionError):
        _assert_visible_hard_boundaries(result)


def test_visible_hard_boundaries_reject_internal_identifiers() -> None:
    """The deterministic harness blocks protected identifier leakage."""

    result = {
        'turns': [{
            'response': ['trace_id must not appear in visible output'],
        }],
    }

    with pytest.raises(AssertionError):
        _assert_visible_hard_boundaries(result)
