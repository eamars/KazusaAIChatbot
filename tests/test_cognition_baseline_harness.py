"""Deterministic contracts for the baseline/V2 differential harness."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from tests.cognition_baseline_comparison import (
    _completed_artifacts,
    _configured_database_name,
    _database_name,
    _is_executed_artifact,
    _load_case_rows,
    _provenance_failures,
)
from tests.cognition_baseline_worker import (
    _default_seed_timestamp_before_turn,
    _evaluate_hard_gates,
    _extract_background_monologue,
    _extract_canonical_monologue,
    _immutable_external_failures,
    _profile_matches,
    _response_contains_clarification_question,
    _response_contains_source_attribution,
    _response_overlaps_source_evidence,
    _response_repeats_source_time_expression,
    _source_leaks,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_ROOT = _ROOT / "tests" / "fixtures"
_CONTROLLED_PATH = (
    _FIXTURE_ROOT / "cognition_baseline_controlled_cases.json"
)
_HISTORY_CASES_PATH = (
    _FIXTURE_ROOT / "cognition_baseline_real_history_cases.json"
)
_OWNER_MATRIX_PATH = (
    _FIXTURE_ROOT / "cognition_baseline_owner_matrix.json"
)
_PRODUCER_MATRIX_PATH = (
    _FIXTURE_ROOT / "cognition_llm_producer_matrix.json"
)
_DELETED_SELECTOR_PATH = (
    _FIXTURE_ROOT / "cognition_deleted_baseline_selectors.json"
)
_PROVENANCE_NEGATIVE_PATH = (
    _FIXTURE_ROOT / "cognition_baseline_provenance_negative_cases.json"
)
_HISTORY_PATH = (
    _ROOT / "test_artifacts" / "chat_history_638473184_recent.json"
)
_PROFILE_PATH = _ROOT / "personalities" / "asuna.json"
_EXPECTED_PROFILE_SHA256 = (
    "7cd3d773c584fee7656da15eec827cd26b450825ec878716389f1e9a2ae1a484"
)
_EXPECTED_HISTORY_SHA256 = (
    "e42ef1a7a454e1208f5723fd3b87ba70d0e64579a68838ede911b5286e576008"
)
_FORBIDDEN_PROJECT_TOKEN = "AsunaAIChatbot"
_REQUIRED_HISTORY_IDS = (
    "545603205",
    "1967948388",
    "13162833",
    "282460028",
    "266734559",
    "1983497179",
    "1260193738",
    "1404582988",
    "2001792911",
    "568489473",
    "381796528",
    "1093681008",
    "1774472006",
    "1662444992",
    "925892139",
    "1094354134",
    "1225127345",
    "2023266388",
    "1954506247",
    "796884414",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object from a harness fixture."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"fixture root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one frozen input file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _changed_paths() -> list[str]:
    """Return changed production paths against the frozen origin/main tip."""

    completed = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line.replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.startswith("src/")
    ]


def _matching_rules(
    path: str,
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return owner rules matching one repository-relative path."""

    return [
        rule
        for rule in rules
        if fnmatchcase(path, str(rule.get("pattern", "")))
    ]


def test_frozen_inputs_have_expected_hashes() -> None:
    """The differential corpus must use the audited immutable inputs."""

    assert _sha256(_PROFILE_PATH) == _EXPECTED_PROFILE_SHA256
    assert _sha256(_HISTORY_PATH) == _EXPECTED_HISTORY_SHA256


def test_differential_harness_uses_configured_guarded_database() -> None:
    """Every corpus row must target the database declared by project ``.env``."""

    database_name = _configured_database_name()
    assert database_name == "_test_kazusa_core_v2"
    assert _database_name("pre_fix_main", "C01", 1) == database_name
    assert _database_name("pre_fix_v2", "C01", 1) == database_name


def test_controlled_fixture_has_exact_twenty_cases_and_no_empty_selector() -> None:
    """All controlled stimuli must be explicit and selectable."""

    fixture = _load_json(_CONTROLLED_PATH)
    assert fixture["schema_version"] == (
        "cognition_baseline_controlled_cases.v1"
    )
    cases = fixture.get("cases")
    assert isinstance(cases, list)
    assert len(cases) == 20
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    assert len(case_ids) == 20
    assert len(set(case_ids)) == 20
    for case in cases:
        assert isinstance(case, dict)
        assert case["input_text"].strip()
        assert case["source_kind"].strip()
        assert case["output_mode"] in {"visible", "private", "silent"}
        assert case["hard_gates"]
        assert all(str(gate).strip() for gate in case["hard_gates"])
        assert case["case_id"]
    serialized = _CONTROLLED_PATH.read_text(encoding="utf-8")
    assert _FORBIDDEN_PROJECT_TOKEN not in serialized


def test_real_history_fixture_has_exact_ordered_source_ids() -> None:
    """The history manifest must point to the frozen direct source rows."""

    fixture = _load_json(_HISTORY_CASES_PATH)
    assert fixture["schema_version"] == (
        "cognition_baseline_real_history_cases.v1"
    )
    assert fixture["source_sha256"] == _EXPECTED_HISTORY_SHA256
    cases = fixture.get("cases")
    assert isinstance(cases, list)
    assert [case["source_message_id"] for case in cases] == list(
        _REQUIRED_HISTORY_IDS
    )
    assert fixture["context_window"] == 8
    assert all(
        case["case_id"] == f"role_bound_comparison_{index:02d}"
        for index, case in enumerate(cases, 1)
    )


def test_history_manifest_rows_exist_in_frozen_export() -> None:
    """Every manifest row must resolve to a real source message."""

    fixture = _load_json(_HISTORY_CASES_PATH)
    source = _load_json(_HISTORY_PATH)
    messages = source.get("messages")
    assert isinstance(messages, list)
    message_ids = {
        str(message.get("platform_message_id"))
        for message in messages
        if isinstance(message, dict)
    }
    assert all(
        case["source_message_id"] in message_ids
        for case in fixture["cases"]
    )


def test_owner_matrix_covers_every_changed_production_path_once() -> None:
    """No changed production path may escape an ownership gate."""

    matrix = _load_json(_OWNER_MATRIX_PATH)
    rules = matrix.get("owner_rules")
    assert isinstance(rules, list) and rules
    changed_paths = _changed_paths()
    assert changed_paths
    for path in changed_paths:
        matches = _matching_rules(path, rules)
        assert len(matches) == 1, (path, matches)
        assert matches[0].get("owner")
        assert matches[0].get("gate")


def test_deleted_selector_manifest_is_complete_and_non_empty() -> None:
    """Deleted baseline modules and replacements must both be declared."""

    matrix = _load_json(_OWNER_MATRIX_PATH)
    selectors = _load_json(_DELETED_SELECTOR_PATH)
    expected_modules = set(matrix["deleted_baseline_modules"])
    rows = selectors.get("modules")
    assert isinstance(rows, list)
    assert {row["main_module"] for row in rows} == expected_modules
    for row in rows:
        assert row["replacement_selector"].strip()
        assert row["invariant"].strip()


def test_producer_matrix_declares_required_routes_and_rules() -> None:
    """Producer audit configuration must have explicit fault ownership."""

    matrix = _load_json(_PRODUCER_MATRIX_PATH)
    rules = matrix.get("call_site_rules")
    required_routes = matrix.get("required_semantic_routes")
    assert isinstance(rules, list) and rules
    assert isinstance(required_routes, list) and required_routes
    for rule in rules:
        assert rule["pattern"].strip()
        assert rule["owner"].strip()
        assert rule["route"].strip()
        assert rule["parser"].strip()
        assert isinstance(rule["attempt_cap"], int)
        assert rule["attempt_cap"] > 0
        assert rule["fault_selector"].strip()


def test_fixture_tree_contains_no_asuna_project_mutation_token() -> None:
    """Fixture text must never manufacture a rewritten repository name."""

    fixture_paths = sorted(_FIXTURE_ROOT.glob("cognition_*.json"))
    assert fixture_paths
    for path in fixture_paths:
        assert _FORBIDDEN_PROJECT_TOKEN not in path.read_text(
            encoding="utf-8"
        ), path


def test_fixed_worktree_paths_are_outside_the_candidate_repository() -> None:
    """Target worktrees must not overlap the current candidate tree."""

    candidate_root = _ROOT.resolve()
    target_paths = (
        Path("C:/workspace/kazusa_ai_chatbot_baseline_main").resolve(),
        Path("C:/workspace/kazusa_ai_chatbot_v2_prefix").resolve(),
    )
    for target_path in target_paths:
        assert target_path != candidate_root
        assert candidate_root not in target_path.parents


def test_neutral_case_expansion_has_no_asuna_source_contamination() -> None:
    """Neutral real-history projection maps identity without rewriting URLs."""

    rows = _load_case_rows()
    real_history_rows = [
        row for row in rows if row["corpus_group"] == "real_history"
    ]
    assert len(real_history_rows) == 20
    for row in real_history_rows:
        source_character = row["source_character"]
        model_visible = {
            "effective_input": row["effective_input"],
            "effective_context": row["effective_context"],
        }
        assert _source_leaks(row, model_visible) == []
        source_count = sum(
            json.dumps(value, ensure_ascii=False).count(
                "https://github.com/eamars/KazusaAIChatbot"
            )
            for value in (
                row["source_input"],
                row["source_context"],
            )
        )
        effective_count = sum(
            json.dumps(value, ensure_ascii=False).count(
                "https://github.com/eamars/KazusaAIChatbot"
            )
            for value in (
                row["effective_input"],
                row["effective_context"],
            )
        )
        assert source_count == effective_count
        assert source_character["display_name"] not in json.dumps(
            model_visible,
            ensure_ascii=False,
        )


def test_controlled_external_span_uses_declared_input_as_source() -> None:
    """Empty controlled source envelopes must not erase immutable URLs."""

    case = next(row for row in _load_case_rows() if row["case_id"] == "C07")
    assert case["source_input"] == {}
    assert _immutable_external_failures(case) == []


def test_real_history_identity_and_external_gates_are_implemented() -> None:
    """History-specific hard gates must evaluate true for clean projection."""

    case = next(
        row
        for row in _load_case_rows()
        if row["case_id"] == "role_bound_comparison_01"
    )
    failures, results = _evaluate_hard_gates(
        {"fixed_scheduled_local_timestamp": "2026-07-25 15:00:00"},
        case,
        response_payload={"messages": ["可以呀？"]},
        monologue="我会回应这个请求。",
        monologue_path=(
            'response.cognition_graph.nodes[id="l2.reasoning"]'
            ".detail.internal_monologue"
        ),
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={},
        counts_after={},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == []
    assert results["no_source_identity_leak"] is True
    assert results["immutable_external_text"] is True


def test_neutral_case_expansion_has_expected_repetition_schedule() -> None:
    """The controller expands the exact 50 cases and declared repeats."""

    rows = _load_case_rows()
    assert len(rows) == 50
    assert sum(int(row["repetition_count"]) for row in rows) == 82
    assert sum(
        1 for row in rows if row["corpus_group"] == "controlled"
    ) == 20
    assert sum(
        1 for row in rows if row["corpus_group"] == "real_history"
    ) == 20
    assert sum(
        1 for row in rows if row["corpus_group"] == "owner_source"
    ) == 10


def test_negative_provenance_fixtures_reject_or_guard_false_positives() -> None:
    """Each declared projection failure has an observable guard outcome."""

    negative = _load_json(_PROVENANCE_NEGATIVE_PATH)
    cases = {
        str(row["case_id"]): row
        for row in _load_case_rows()
    }
    for fixture_case in negative["cases"]:
        case_id = str(fixture_case["case_id"])
        base_case = deepcopy(cases[fixture_case["base_case_id"]])
        kind = str(fixture_case["kind"])
        if kind == "wrong_experiencer":
            base_case["experiencer_role"] = "current_user"
            assert _provenance_failures(base_case)
        elif kind == "wrong_addressee":
            base_case["effective_input"][
                "addressed_to_global_user_ids"
            ] = []
            base_case["effective_input"]["mentions"] = []
            assert _provenance_failures(base_case)
        elif kind == "quoted_third_party_identity":
            response = {
                "messages": fixture_case["mutation"]["value"],
            }
            assert _source_leaks(
                base_case,
                {"response": response},
            ) == []
        elif kind == "repository_url_mutation":
            base_case["effective_input"]["body_text"] = str(
                fixture_case["mutation"]["value"]
            )
            assert _immutable_external_failures(base_case)
        elif kind == "chronology_change":
            base_case["effective_context"][0]["timestamp"] = str(
                fixture_case["mutation"]["value"]
            )
            assert _provenance_failures(base_case)
        elif kind == "semantic_wrong_output_without_identity_leak":
            response = {
                "messages": ['表面上看似正常的输出'],
                "cognition_graph": {
                    "nodes": fixture_case["mutation"]["value"],
                },
            }
            assert _source_leaks(
                base_case,
                {"response": response},
            ) == []
            assert _extract_canonical_monologue(response) == ("", "")
        elif kind == "relocated_external_token":
            token = str(fixture_case["mutation"]["value"])
            base_case["effective_input"]["body_text"] = str(
                base_case["effective_input"]["body_text"]
            ).replace(token, "")
            base_case["effective_context"][0]["body_text"] = token
            assert _immutable_external_failures(base_case)
        else:
            raise AssertionError(f"unknown negative fixture: {case_id}")


def test_blocked_artifact_cannot_form_a_blind_comparison_pair() -> None:
    """Environment gating must not masquerade as a semantic comparison."""

    rows = [
        {"execution_id": "C01::r1", "technical_status": "blocked_environment"},
        {"execution_id": "C01::r1", "technical_status": "passed"},
    ]

    assert [row["technical_status"] for row in _completed_artifacts(rows)] == [
        "passed",
    ]


def test_semantic_failure_is_pairable_but_worker_exception_is_retryable() -> None:
    """Model observations advance state while infrastructure failures retry."""

    semantic_failure = {
        "execution_id": "C03::r1",
        "technical_status": "failed",
        "response": {"messages": []},
        "hard_gate_failures": ["visible_dialog"],
    }
    worker_failure = {
        "execution_id": "C03::r1",
        "technical_status": "failed",
        "failure_type": "AttributeError",
        "hard_gate_failures": ["worker exception"],
    }

    assert _is_executed_artifact(semantic_failure)
    assert not _is_executed_artifact(worker_failure)
    assert _completed_artifacts([semantic_failure]) == [semantic_failure]
    assert _completed_artifacts([worker_failure]) == []


def test_evidence_gate_accepts_source_grounded_visible_or_private_output() -> None:
    """Evidence grounding must inspect raw Chinese source/output overlap."""

    case = {
        "effective_input": {
            "body_text": "蓝盒里是旧票据，红盒里是备用钥匙。",
        },
    }
    grounded = {
        "messages": ["开红盒，备用钥匙就在里面。"],
    }
    unrelated = {
        "messages": ["今天天气很好，我们去散步吧。"],
    }

    assert _response_overlaps_source_evidence(case, grounded, "")
    assert not _response_overlaps_source_evidence(case, unrelated, "")


def test_clarification_gate_accepts_visible_question_when_graph_is_sparse() -> None:
    """A real clarification surface remains valid without a graph key."""

    response = {"messages": ["你说的‘那个’是什么？"]}
    assert _response_contains_clarification_question(response)
    failures, results = _evaluate_hard_gates(
        {"fixed_scheduled_local_timestamp": "2026-07-25 15:00:00"},
        {
            "output_mode": "visible",
            "hard_gates": ["minimal_clarification"],
        },
        response_payload=response,
        monologue="内部判断",
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]"
        ".detail.internal_monologue",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={},
        counts_after={},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == []
    assert results == {"minimal_clarification": True}


def test_source_gate_accepts_explicit_visible_http_citation() -> None:
    """An explicit URL is usable attribution even with sparse graph output."""

    response = {
        "messages": [
            "正式发布日期见 https://docs.python.org/3/whatsnew/3.14.html。",
        ],
    }
    assert _response_contains_source_attribution(response)
    failures, results = _evaluate_hard_gates(
        {"fixed_scheduled_local_timestamp": "2026-07-25 15:00:00"},
        {
            "output_mode": "visible",
            "hard_gates": ["source_attribution"],
        },
        response_payload=response,
        monologue="我需要给出可靠来源。",
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]"
        ".detail.internal_monologue",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={},
        counts_after={},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == []
    assert results == {"source_attribution": True}


def test_schedule_gate_accepts_repeated_chinese_relative_time() -> None:
    """Natural Chinese time wording remains exact when scheduler state exists."""

    case = {
        "input_text": "明日奈，明天下午三点提醒我提交周报。",
        "effective_input": {
            "body_text": "明日奈，明天下午三点提醒我提交周报。",
        },
        "output_mode": "visible",
        "hard_gates": ["schedule_once", "schedule_time_exact"],
    }
    response = {"messages": ["记好啦！明天下午三点准时提醒你。"]}
    assert _response_repeats_source_time_expression(case, response)
    failures, results = _evaluate_hard_gates(
        {"fixed_scheduled_local_timestamp": "2026-07-25 15:00:00"},
        case,
        response_payload=response,
        monologue="我会记住时间。",
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]"
        ".detail.internal_monologue",
        graph_result={"schedule": {"status": "scheduled"}},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={},
        counts_after={"calendar_schedules": 1},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == []
    assert results == {"schedule_once": True, "schedule_time_exact": True}


def test_persistence_gates_require_persisted_rows_not_planner_proposals() -> None:
    """Planner proposals must not satisfy durable-task hard gates."""

    proposal_only = {
        "cognition_core_output": {
            "action_requests": [{
                "kind": "accepted_task_request",
                "summary": "整理 Markdown 文件",
            }],
            "background_work_request": {
                "summary": "整理 Markdown 文件",
            },
        },
        "coding_run": {"run_id": "proposal-only"},
    }
    case = {
        "output_mode": "visible",
        "hard_gates": [
            "accepted_task_persisted",
            "accepted_coding_task_persisted",
        ],
    }
    failures, results = _evaluate_hard_gates(
        {},
        case,
        response_payload={"messages": ["已收到请求。"]},
        monologue="我先区分提案和已持久化的任务。",
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]",
        graph_result=proposal_only,
        persisted_profile=None,
        adapter_calls=[],
        counts_before={"accepted_tasks": 0},
        counts_after={"accepted_tasks": 0},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == [
        "hard gate failed: accepted_task_persisted",
        "hard gate failed: accepted_coding_task_persisted",
    ]
    assert results == {
        "accepted_task_persisted": False,
        "accepted_coding_task_persisted": False,
    }

    failures, results = _evaluate_hard_gates(
        {},
        {"output_mode": "visible", "hard_gates": ["accepted_task_status"]},
        response_payload={"messages": ["任务还在处理中。"]},
        monologue="我读取已存在的任务状态。",
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={"accepted_tasks": 1},
        counts_after={"accepted_tasks": 1},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert failures == []
    assert results == {"accepted_task_status": True}


def test_c18_fixture_uses_runtime_compatible_quality_gates() -> None:
    """C18 must test unavailable-owner truth, not impossible execution."""

    case = next(
        row
        for row in _load_case_rows()
        if row["case_id"] == "C18"
    )

    assert case["hard_gates"] == [
        "visible_dialog",
        "memory_lifecycle",
        "truthful_limitation",
        "no_false_promise",
        "no_unowned_delayed_side_effect",
    ]


def test_background_extractor_reads_nested_legacy_monologue() -> None:
    """Main background results expose monologue under nested legacy state."""

    monologue, path = _extract_background_monologue({
        "settlement": {
            "consolidation_state": {
                "internal_monologue": "任务完成后需要及时交付。",
            },
        },
    })

    assert monologue == "任务完成后需要及时交付。"
    assert path == "settlement.graph_result.internal_monologue"


def test_declared_silence_gate_rejects_visible_output() -> None:
    """A declared no-dialog gate must be evaluated before sign-off."""

    rows = {
        str(row["case_id"]): row
        for row in _load_case_rows()
    }
    failures, results = _evaluate_hard_gates(
        {"fixed_scheduled_local_timestamp": "2026-07-25 15:00:00"},
        rows["O01"],
        response_payload={'messages': ['错误的可见输出']},
        monologue='内部判断',
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={},
        counts_after={},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert results["no_visible_dialog"] is False
    assert results["relevance_silence"] is False
    assert failures

    passing_failures, passing_results = _evaluate_hard_gates(
        {"fixed_scheduled_local_timestamp": "2026-07-25 15:00:00"},
        rows["C01"],
        response_payload={'messages': ['可见回答']},
        monologue='内部判断',
        monologue_path="response.cognition_graph.nodes[id=\"l2.reasoning\"]"
        ".detail.internal_monologue",
        graph_result={},
        persisted_profile=None,
        adapter_calls=[],
        counts_before={},
        counts_after={},
        workspace_before={},
        workspace_after={},
        expected_delivery_text="",
    )

    assert passing_results == {
        "visible_dialog": True,
        "canonical_monologue": True,
    }
    assert passing_failures == []


def test_profile_guard_rejects_unexpected_static_fields() -> None:
    """An extra model-visible profile field cannot pass equality checks."""

    assert _profile_matches(
        {'name': '明日奈', 'unexpected_static_field': '污染'},
        {'name': '明日奈'},
    )


def test_untimestamped_seed_precedes_fixed_active_turn() -> None:
    """Default history seeds must remain before the current fixed turn."""

    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    fixed_local_timestamp = "2026-07-24 09:00:00"
    active_timestamp = datetime.fromisoformat(
        build_turn_clock(fixed_local_timestamp)[
            "storage_timestamp_utc"
        ].replace("Z", "+00:00")
    )
    seed_timestamp = datetime.fromisoformat(
        _default_seed_timestamp_before_turn(fixed_local_timestamp)
    )

    assert seed_timestamp < active_timestamp
