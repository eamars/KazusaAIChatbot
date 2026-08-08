"""Deterministic contracts for the Asuna private R18 full E2E harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_asuna_private_r18_affinity_live_llm import (
    _ARTIFACT_ROOT,
    _DEFAULT_RELATIONSHIP_VALUES,
    _MANIFEST_PATH,
    _RELATIONSHIP_FIELDS,
    _TEST_DATABASE_NAME,
    _guarded_path,
    _input_sequence,
    _load_manifest,
    _manifest_cases,
    _relationship_seed_values,
)
from tests import run_asuna_private_r18_affinity_replay as replay_controller
from tests import test_asuna_private_r18_affinity_live_llm as live_harness

requires_private_manifest = pytest.mark.skipif(
    not _MANIFEST_PATH.is_file(),
    reason="private R18 replay manifest is unavailable",
)


@requires_private_manifest
def test_private_r18_manifest_is_the_exact_20_input_sequence() -> None:
    """The harness selects only the corrected private R18 user inputs."""

    manifest = _load_manifest()
    cases = _manifest_cases(manifest)
    inputs = _input_sequence(manifest)
    assert manifest["scenario"] == "private_r18"
    assert len(cases) == 20
    assert len(inputs) == 20
    assert [item["case_index"] for item in inputs] == list(range(1, 21))
    assert set(inputs[0]) == {
        "case_index",
        "platform_message_id",
        "body_text",
        "timestamp",
    }
    assert Path(_MANIFEST_PATH).is_file()


@requires_private_manifest
def test_full_sequence_does_not_project_old_dialog_or_residue() -> None:
    """The full E2E input projection excludes prior R18 state artifacts."""

    manifest = _load_manifest()
    for input_row in _input_sequence(manifest):
        assert "old_dialog" not in input_row
        assert "old_residue" not in input_row
    for case in _manifest_cases(manifest):
        source_message = case["source_message"]
        assert source_message["body_text"] == next(
            item["body_text"]
            for item in _input_sequence(manifest)
            if item["case_index"] == case["case_index"]
        )


@requires_private_manifest
def test_high_and_default_relationship_seeds_are_explicitly_distinct() -> None:
    """The two sequences differ at the native relationship seed boundary."""

    manifest = _load_manifest()
    high = _relationship_seed_values("high_affinity", manifest)
    default = _relationship_seed_values("default_affinity", manifest)
    assert set(high) == set(_RELATIONSHIP_FIELDS)
    assert default == _DEFAULT_RELATIONSHIP_VALUES
    assert high != default
    assert high["familiarity"] == 95
    assert high["trust"] == 90
    assert high["boundary_safety"] == 85


def test_full_e2e_contract_uses_guarded_empty_baseline() -> None:
    """The declared baseline contains identity and affinity only."""

    assert _TEST_DATABASE_NAME == "_test_kazusa_live_llm"
    assert _ARTIFACT_ROOT.name == "asuna_private_r18_affinity_replay"


def test_artifact_paths_are_guarded_to_the_replay_root(tmp_path: Path) -> None:
    """The live child cannot redirect evidence outside test_artifacts."""

    inside = _guarded_path(
        _ARTIFACT_ROOT / "contract" / "turn.json",
    )
    assert inside.parent == (_ARTIFACT_ROOT / "contract").resolve()
    with pytest.raises(ValueError, match="escaped"):
        _guarded_path(tmp_path / "outside.json")


@requires_private_manifest
def test_replay_request_uses_service_time_and_keeps_source_provenance() -> None:
    """Model-facing rows use runtime chronology while the fixture keeps time."""

    manifest = _load_manifest()
    case = _manifest_cases(manifest)[0]
    source = case["source_message"]
    request = live_harness._build_request(
        case=case,
        runtime={
            "platform": str(source["platform"]),
            "channel_id": str(source["platform_channel_id"]),
        },
    )

    assert request.local_timestamp == ""
    assert _input_sequence(manifest)[0]["timestamp"] == source["timestamp"]


def test_replay_history_rejects_split_user_assistant_chronology() -> None:
    """An assistant must immediately follow the user row owning its trace."""

    split_rows = [
        {
            "role": "user",
            "llm_trace_id": "trace-1",
            "timestamp": "2026-07-17T00:00:00Z",
        },
        {
            "role": "user",
            "llm_trace_id": "trace-2",
            "timestamp": "2026-07-17T00:01:00Z",
        },
        {
            "role": "assistant",
            "llm_trace_id": "trace-1",
            "timestamp": "2026-07-25T00:00:00Z",
        },
        {
            "role": "assistant",
            "llm_trace_id": "trace-2",
            "timestamp": "2026-07-25T00:01:00Z",
        },
    ]
    with pytest.raises(
        live_harness.FatalSequenceError,
        match="chronology",
    ):
        live_harness._assert_conversation_chronology(split_rows)

    live_harness._assert_conversation_chronology([
        split_rows[0],
        split_rows[2],
        split_rows[1],
        split_rows[3],
    ])


def test_replay_child_requires_full_trace_capture() -> None:
    """Every future replay child retains protected model inputs and outputs."""

    output_root = _ARTIFACT_ROOT / "contract" / "full-trace"
    environment = replay_controller._child_environment(
        run_id="contract",
        condition="high_affinity",
        output_root=output_root,
    )

    assert environment["LLM_TRACE_CAPTURE_MODE"] == "full"
    live_harness._assert_full_trace_capture([{
        "stage_name": "message_decontextualizer",
        "raw_messages": [{"role": "system", "content": "contract"}],
        "raw_response_text": '{"output": "result"}',
        "parsed_output": {"output": "result"},
    }])
    with pytest.raises(
        live_harness.FatalSequenceError,
        match="full trace",
    ):
        live_harness._assert_full_trace_capture([{
            "stage_name": "message_decontextualizer",
            "raw_messages": [],
            "raw_response_text": "",
            "parsed_output": {},
        }])


def test_replay_controller_emits_data_only() -> None:
    """The controller leaves readable quality judgment to the reviewing agent."""

    assert not hasattr(replay_controller, "_render_review")
    parser = replay_controller._build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert "review" not in choices
