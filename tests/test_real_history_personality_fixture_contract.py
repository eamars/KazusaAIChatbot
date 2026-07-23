"""Deterministic contracts for the real-history personality fixture."""

from __future__ import annotations

from pathlib import Path

from tests.test_real_history_personality_e2e_live_llm import (
    _CASES,
    _build_profile_case,
    _build_request,
    _fixture_validity,
    _load_json_object,
    _relevance_dispositions,
    _source_identity_tokens,
)


_ROOT = Path(__file__).resolve().parents[1]


def _profile_name(label: str) -> str:
    """Load one profile name without relying on runtime configuration."""

    profile = _load_json_object(_ROOT / "personalities" / f"{label}.json")
    return str(profile["name"])


def test_fixture_has_twenty_direct_kazusa_history_rows() -> None:
    """The paired population must contain only direct source rows."""

    assert len(_CASES) == 20
    assert len({case["source_index"] for case in _CASES}) == 20
    assert all(
        case["category"] == "direct_kazusa_history"
        for case in _CASES
    )
    assert all(
        case["source_character"]["display_name"] == "杏山千纱"
        for case in _CASES
    )


def test_asuna_projection_removes_source_identity_from_every_case() -> None:
    """Asuna must receive mapped identity evidence for every paired case."""

    for case in _CASES:
        profile_case = _build_profile_case(
            case=case,
            profile_label="asuna",
            profile_name=_profile_name("asuna"),
            character_global_user_id="character-global",
            current_global_id="real-history-current-global",
        )
        request = _build_request(
            profile_case=profile_case,
            character_global_user_id="character-global",
            channel_id="fixture-contract",
            platform_message_id="fixture-current",
        )
        validity = _fixture_validity(
            profile_case=profile_case,
            profile_label="asuna",
            request=request,
        )
        assert validity["passed"], validity
        leaks = _collect_source_leaks_for_test(profile_case, request)
        assert leaks == [], (case["case_id"], leaks)
        assert (
            profile_case["effective_input"]["body_text"]
            != profile_case["source_input"]["body_text"]
            or "杏山千纱" not in profile_case["source_input"]["body_text"]
        )


def test_kazusa_projection_preserves_source_body_text() -> None:
    """Kazusa remains the unchanged source-language comparison baseline."""

    for case in _CASES:
        profile_case = _build_profile_case(
            case=case,
            profile_label="kazusa",
            profile_name=_profile_name("kazusa"),
            character_global_user_id="character-global",
            current_global_id="real-history-current-global",
        )
        assert (
            profile_case["effective_input"]["body_text"]
            == profile_case["source_input"]["body_text"]
        )
        request = _build_request(
            profile_case=profile_case,
            character_global_user_id="character-global",
            channel_id="fixture-contract",
            platform_message_id="fixture-current",
        )
        validity = _fixture_validity(
            profile_case=profile_case,
            profile_label="kazusa",
            request=request,
        )
        assert validity["passed"], validity


def test_routing_guard_has_no_typed_active_target() -> None:
    """The separate guard must not recreate the contaminated envelope."""

    profile_case = _build_profile_case(
        case=_CASES[0],
        profile_label="asuna",
        profile_name=_profile_name("asuna"),
        character_global_user_id="character-global",
        current_global_id="real-history-current-global",
        routing_guard=True,
    )
    request = _build_request(
        profile_case=profile_case,
        character_global_user_id="character-global",
        channel_id="routing-guard-contract",
        platform_message_id="routing-guard-current",
    )
    envelope = request.message_envelope.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    assert envelope["mentions"] == []
    assert envelope["addressed_to_global_user_ids"] == []
    validity = _fixture_validity(
        profile_case=profile_case,
        profile_label="asuna",
        request=request,
    )
    assert validity["passed"], validity


def test_routing_guard_reads_frontline_discard_disposition() -> None:
    """Frontline discard must prevent the old relevance false positive."""

    trace_steps = [
        {
            "stage_name": "frontline_relevance_agent",
            "parsed_output": {
                "intake_action": "discard",
            },
        },
    ]

    assert _relevance_dispositions(trace_steps) == ["discard"]


def _collect_source_leaks_for_test(
    profile_case: dict,
    request: object,
) -> list[dict[str, str]]:
    """Keep the assertion helper local to deterministic fixture tests."""

    from tests.test_real_history_personality_e2e_live_llm import (
        _collect_source_leaks,
    )

    model_visible = {
        "effective_input": profile_case["effective_input"],
        "effective_context": profile_case["effective_context"],
        "effective_envelope": request.message_envelope.model_dump(
            exclude_none=True,
            exclude_defaults=True,
        ),
    }
    return _collect_source_leaks(
        model_visible,
        source_tokens=_source_identity_tokens(
            profile_case["source_character"]
        ),
    )
