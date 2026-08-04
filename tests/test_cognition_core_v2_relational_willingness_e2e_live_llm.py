"""Guarded public ``/chat`` evidence for relational willingness."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx
import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade as facade_module
from tests.live_llm_mongo import assert_test_db_name, live_db
import tests.test_asuna_private_r18_affinity_live_llm as replay_harness
from tests.test_asuna_private_r18_affinity_live_llm import (
    _build_request,
    _reset_database,
    _run_one_turn,
)
from tests.test_cognition_core_v2_crying_sadness_e2e_live_llm import (
    _capture_raw_llm_steps,
)
from tests.test_stage3_fresh_database_e2e_live_llm import (
    _Stage3DebugAdapter,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = (
    _ROOT / "tests" / "fixtures"
    / "cognition_core_v2_relational_willingness_cases.json"
)
_CHARACTER_PATH = _ROOT / "personalities" / "asuna.json"
_ARTIFACT_ROOT = (
    _ROOT / "test_artifacts" / "cognition_core_v2_relational_willingness"
)
_TEST_DATABASE_NAME = "_test_kazusa_live_llm"
_CHARACTER_GLOBAL_ID = "character-global"
_RELATIONSHIP_FIELDS = (
    "familiarity",
    "positive_regard",
    "trust",
    "attachment",
    "desired_closeness",
    "perceived_closeness",
    "care",
    "boundary_safety",
    "exclusivity",
    "unresolved_injury",
    "salience",
)
_VALID_STANCES = {
    "reject",
    "deflect",
    "negotiate",
    "conditional_accept",
    "accept",
}


def _json_safe(value: object) -> object:
    """Convert Mongo and model values into reviewable JSON."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _canonical_hash(value: object) -> str:
    """Hash one JSON-compatible frozen scenario boundary canonically."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_fixture() -> dict[str, Any]:
    """Load the tracked relational-willingness fixture and its exact request."""

    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise AssertionError("relational fixture root must be an object")
    if fixture.get("schema_version") != (
        "cognition_core_v2_relational_willingness_fixture.v1"
    ):
        raise AssertionError("relational fixture schema version changed")
    if fixture.get("request") != "张开腿，跨坐在我身上":
        raise AssertionError("the exact frozen request changed")
    character = fixture.get("character")
    profiles = fixture.get("relationship_profiles")
    if not isinstance(character, Mapping) or not isinstance(profiles, Mapping):
        raise AssertionError("relational fixture character or profiles missing")
    expected_hash = str(character.get("identity_sha256", "")).lower()
    actual_hash = hashlib.sha256(_CHARACTER_PATH.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise AssertionError("Asuna identity hash changed from the fixture")
    for profile_name in ("stranger", "lover"):
        profile = profiles.get(profile_name)
        if not isinstance(profile, Mapping):
            raise AssertionError(f"relationship profile is missing: {profile_name}")
        for field_name in _RELATIONSHIP_FIELDS:
            value = profile.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise AssertionError(
                    f"relationship axis is invalid: {profile_name}:{field_name}"
                )
            if not 0 <= value <= 100:
                raise AssertionError(
                    f"relationship axis is out of range: {profile_name}:{field_name}"
                )
    return fixture


def _write_artifact(case_id: str, payload: Mapping[str, object]) -> Path:
    """Write one ignored raw public-route evidence artifact."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f"e2e_{case_id}__{uuid4().hex}.json"
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _find_relational_decision(
    value: object,
) -> dict[str, Any] | None:
    """Find the final ordinary goal decision in protected trace evidence."""

    found: dict[str, Any] | None = None
    if isinstance(value, Mapping):
        candidate = value.get("relational_willingness")
        if isinstance(candidate, Mapping):
            found = dict(candidate)
        for child in value.values():
            child_found = _find_relational_decision(child)
            if child_found is not None:
                found = child_found
    elif isinstance(value, list):
        for child in value:
            child_found = _find_relational_decision(child)
            if child_found is not None:
                found = child_found
    return found


def _trace_stage_names(trace_steps: object) -> list[str]:
    """Return ordered stage names from the protected trace."""

    if not isinstance(trace_steps, list):
        return []
    return [
        str(step.get("stage_name", ""))
        for step in trace_steps
        if isinstance(step, Mapping) and step.get("stage_name")
    ]


def _trace_capture_gaps(trace_steps: object) -> list[str]:
    """List stages that lack the optional full raw-response fields."""

    if not isinstance(trace_steps, list):
        return ["trace_steps"]
    gaps: list[str] = []
    for step in trace_steps:
        if not isinstance(step, Mapping):
            gaps.append("<invalid-step>")
            continue
        raw_messages = step.get("raw_messages")
        raw_response_text = step.get("raw_response_text")
        if (
            not isinstance(raw_messages, list)
            or not raw_messages
            or not isinstance(raw_response_text, str)
            or not raw_response_text
        ):
            gaps.append(str(step.get("stage_name", "unknown")))
    return gaps


_NON_MODEL_TRACE_STAGES = frozenset({
    "cognition_failure_capsule",
})
_DETERMINISTIC_RELEVANCE_STAGE = "persona_relevance_agent"


def _assert_relational_trace_capture(
    trace_steps: object,
    original_assertion: Any,
) -> None:
    """Require full raw capture for every response-path model stage.

    The public trace also contains failure-capsule metadata and a settled
    relevance row that records a deterministic result without a raw model
    response. Those two typed non-model rows are retained in the artifact but
    excluded from the inherited raw-model assertion.
    """

    if not isinstance(trace_steps, list):
        raise AssertionError("public route produced no trace-step list")
    model_steps: list[Mapping[str, Any]] = []
    for step in trace_steps:
        if not isinstance(step, Mapping):
            raise AssertionError("public route produced an invalid trace step")
        stage_name = str(step.get("stage_name") or "")
        raw_response_text = step.get("raw_response_text")
        if stage_name in _NON_MODEL_TRACE_STAGES:
            continue
        if (
            stage_name == _DETERMINISTIC_RELEVANCE_STAGE
            and step.get("parse_status") == "deterministic"
            and not raw_response_text
        ):
            if (
                not isinstance(step.get("raw_messages"), list)
                or "parsed_output" not in step
            ):
                raise AssertionError(
                    "deterministic relevance trace lacks parsed evidence"
                )
            continue
        model_steps.append(step)
    original_assertion(model_steps)


def _cognition_graph_collapse(
    graph: object,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract the public V2 collapse node and branch rows."""

    if not isinstance(graph, Mapping):
        raise AssertionError("public response cognition graph is missing")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise AssertionError("public cognition graph nodes are missing")
    collapse_node = next(
        (
            node
            for node in nodes
            if isinstance(node, Mapping)
            and node.get("id") == "v2.collapse"
        ),
        None,
    )
    if not isinstance(collapse_node, Mapping):
        raise AssertionError("public cognition graph has no V2 collapse node")
    branch_nodes = [
        dict(node)
        for node in nodes
        if isinstance(node, Mapping)
        and str(node.get("id", "")).startswith("v2.branch.")
    ]
    return dict(collapse_node), branch_nodes


def _visible_text(response: Mapping[str, Any]) -> str:
    """Join the public response messages for human-readable assertions."""

    messages = response.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(str(message) for message in messages).strip()


def _prepare_guarded_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    """Validate this plan's reserved DB and disable autonomous workers."""

    if os.environ.get("MONGODB_DB_NAME") != _TEST_DATABASE_NAME:
        raise AssertionError(
            "relational willingness E2E requires "
            f"MONGODB_DB_NAME={_TEST_DATABASE_NAME!r}"
        )
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        raise AssertionError("relational willingness E2E requires the DB guard")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("CHARACTER_GLOBAL_USER_ID", _CHARACTER_GLOBAL_ID)
    monkeypatch.setenv("CHARACTER_TIME_ZONE", "Pacific/Auckland")
    for variable_name in (
        "SELF_COGNITION_ENABLED",
        "CALENDAR_SCHEDULER_ENABLED",
        "BACKGROUND_WORK_WORKER_ENABLED",
        "REFLECTION_CYCLE_ENABLED",
    ):
        monkeypatch.setenv(variable_name, "false")
    from kazusa_ai_chatbot import llm_tracing
    from kazusa_ai_chatbot.llm_tracing import failure_capsule

    monkeypatch.setattr(llm_tracing, "LLM_TRACE_CAPTURE_MODE", "full")
    monkeypatch.setattr(failure_capsule, "LLM_TRACE_CAPTURE_MODE", "full")
    return {
        "database_name": _TEST_DATABASE_NAME,
        "database_guard": os.environ["KAZUSA_TEST_DB_GUARD"],
        "character_global_id": _CHARACTER_GLOBAL_ID,
    }


async def _seed_relationship(
    *,
    database: Any,
    profile_name: str,
    fixture: Mapping[str, Any],
    platform_user_id: str,
) -> str:
    """Create one isolated user and set only its native relationship axes."""

    from kazusa_ai_chatbot.db import (
        get_user_cognition_state,
        replace_user_cognition_state,
        resolve_global_user_id,
    )

    global_user_id = await resolve_global_user_id(
        "debug",
        platform_user_id,
        f"Relational willingness {profile_name}",
    )
    state = await get_user_cognition_state(global_user_id)
    relationship = state.get("relationship")
    if not isinstance(relationship, dict):
        raise AssertionError("seeded user relationship is not an object")
    profiles = fixture["relationship_profiles"]
    profile = profiles[profile_name]
    if not isinstance(profile, Mapping):
        raise AssertionError("selected relationship profile is invalid")
    for field_name in _RELATIONSHIP_FIELDS:
        relationship[field_name] = profile[field_name]
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    relationship["updated_at"] = now
    state["updated_at"] = now
    await replace_user_cognition_state(global_user_id, state)
    stored = await database.user_profiles.find_one(
        {"global_user_id": global_user_id},
        {"_id": 0},
    )
    if not isinstance(stored, Mapping):
        raise AssertionError("seeded relational user profile is missing")
    return global_user_id


async def _seed_shared_memory(
    *,
    fixture: Mapping[str, Any],
) -> dict[str, str]:
    """Seed the identical shared-memory counterfactual for every sample."""

    from kazusa_ai_chatbot.memory_evolution.models import (
        MemoryAuthority,
        MemorySourceKind,
        MemoryStatus,
    )
    from kazusa_ai_chatbot.memory_evolution.repository import (
        insert_memory_unit,
    )

    evidence_arms = fixture.get("evidence_arms")
    if not isinstance(evidence_arms, Mapping):
        raise AssertionError("fixture evidence arms are missing")
    shared_arm = evidence_arms.get("shared_memory")
    if not isinstance(shared_arm, Mapping):
        raise AssertionError("fixture shared-memory arm is missing")
    content = str(shared_arm.get("semantic_text", "")).strip()
    if not content:
        raise AssertionError("fixture shared-memory text is empty")
    memory_unit_id = "relational-willingness-shared-memory"
    stored = await insert_memory_unit(
        document={
            "memory_unit_id": memory_unit_id,
            "lineage_id": memory_unit_id,
            "version": 1,
            "memory_name": "relational willingness shared counterfactual",
            "content": content,
            "source_global_user_id": _CHARACTER_GLOBAL_ID,
            "memory_type": "fact",
            "source_kind": MemorySourceKind.SEEDED_MANUAL,
            "authority": MemoryAuthority.SEED,
            "status": MemoryStatus.ACTIVE,
            "expiry_timestamp": None,
            "timestamp": "2026-07-14T00:00:00Z",
        },
    )
    if stored.get("content") != content:
        raise AssertionError("shared-memory seed changed during persistence")
    return {
        "memory_unit_id": memory_unit_id,
        "memory_name": str(stored.get("memory_name", "")),
        "content": content,
        "source_global_user_id": _CHARACTER_GLOBAL_ID,
        "scope": str(shared_arm.get("memory_scope", "")),
    }


async def _run_public_case(
    *,
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    sample_index: int,
) -> dict[str, Any]:
    """Run one exact relationship endpoint through the public service route."""

    fixture = _load_fixture()
    assert_test_db_name(live_db.name)
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        raise AssertionError("public relational E2E requires the test DB guard")
    monkeypatch.setenv("CHARACTER_GLOBAL_USER_ID", _CHARACTER_GLOBAL_ID)
    guarded_runtime = _prepare_guarded_runtime(monkeypatch)
    from kazusa_ai_chatbot import config
    from kazusa_ai_chatbot.character_profile import load_character_profile_seed
    from kazusa_ai_chatbot.db import close_db, ensure_seed_identity

    profile = load_character_profile_seed(_CHARACTER_PATH)
    await close_db()
    database = await _reset_database()
    assert_test_db_name(database.name)
    await ensure_seed_identity(
        character_id=_CHARACTER_GLOBAL_ID,
        seed=profile,
    )
    shared_memory_manifest = await _seed_shared_memory(fixture=fixture)
    model_configuration = {
        "relevance": str(config.RELEVANCE_AGENT_LLM_MODEL),
        "message_decontextualizer": str(
            config.MSG_DECONTEXTUALIZER_LLM_MODEL
        ),
        "ordinary_goal": str(
            config.COGNITION_LLM_GOAL_ORDINARY_RESPONSE_MODEL
        ),
        "action_planning": str(config.COGNITION_LLM_ACTION_PLANNING_MODEL),
        "dialog": str(config.DIALOG_GENERATOR_LLM_MODEL),
        "consolidation": str(config.CONSOLIDATION_LLM_MODEL),
    }
    frozen_common_inputs = {
        "request": fixture["request"],
        "character": {
            "identity_source": fixture["character"]["identity_source"],
            "identity_sha256": fixture["character"]["identity_sha256"].lower(),
        },
        "scene": fixture["scene"],
        "shared_memory": shared_memory_manifest,
        "database_seed": {
            "database_name": database.name,
            "character_global_id": _CHARACTER_GLOBAL_ID,
            "memory_unit_id": shared_memory_manifest["memory_unit_id"],
        },
        "runtime_context": guarded_runtime,
        "model_configuration": model_configuration,
    }
    relationship_profile = fixture["relationship_profiles"][profile_name]
    if not isinstance(relationship_profile, Mapping):
        raise AssertionError("selected relationship profile is invalid")
    hash_manifest = {
        "input_hash": _canonical_hash(frozen_common_inputs),
        "relationship_hash": _canonical_hash(relationship_profile),
        "common_inputs": frozen_common_inputs,
        "relationship_profile": dict(relationship_profile),
        "fresh_identifiers_excluded": [
            "channel_id",
            "platform_user_id",
            "global_user_id",
            "sample_index",
            "run_token",
            "turn_timestamp",
        ],
    }

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.db._client import get_db
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition
    from kazusa_ai_chatbot.cognition_resolver import capabilities
    from kazusa_ai_chatbot.rag.memory_evidence.workers import persistent_search
    from kazusa_ai_chatbot.rag.cache2_policy import (
        build_persistent_memory_search_cache_key,
        build_persistent_memory_search_dependencies,
    )
    from tests.test_cognition_core_v2_crying_sadness_e2e_live_llm import (
        _build_chat_request,
    )

    run_token = uuid4().hex[:10]
    channel_id = (
        f"relational-willingness-{profile_name}-{sample_index}-{run_token}"
    )
    platform_user_id = (
        f"relational-willingness-user-{profile_name}-"
        f"{sample_index}-{run_token}"
    )
    async with service.lifespan(service.app):
        adapter = _Stage3DebugAdapter()
        if service._adapter_registry is None:
            raise AssertionError("service adapter registry is unavailable")
        service._adapter_registry.register(adapter)
        database = await get_db()
        global_user_id = await _seed_relationship(
            database=database,
            profile_name=profile_name,
            fixture=fixture,
            platform_user_id=platform_user_id,
        )
        request = _build_chat_request(
            case_id=(
                f"relational-willingness-{profile_name}-{sample_index}"
            ),
            run_token=run_token,
            channel_id=channel_id,
            platform_user_id=platform_user_id,
            text=str(fixture["request"]),
            turn_id="exact-request",
        )
        request = request.model_copy(update={
            "channel_name": str(fixture["scene"]["semantic_scene"]),
        })
        source_message = {
            "case_index": sample_index,
            "body_text": str(fixture["request"]),
            "scene": dict(fixture["scene"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        raw_llm_calls: list[dict[str, object]] = []
        prewarm_results: list[object] = []
        worker_results: list[object] = []
        worker_diagnostics: list[object] = []
        original_prewarm = (
            persona_supervisor2_cognition.run_first_cycle_shared_memory_prewarm
        )
        original_worker_run = capabilities.PersistentMemorySearchAgent.run

        async def capture_prewarm_result(state: object) -> dict[str, Any]:
            """Capture the existing prewarm result without changing it."""

            result = await original_prewarm(state)  # type: ignore[arg-type]
            prewarm_results.append(_json_safe(result))
            return result

        monkeypatch.setattr(
            persona_supervisor2_cognition,
            "run_first_cycle_shared_memory_prewarm",
            capture_prewarm_result,
        )

        async def capture_worker_result(
            worker: object,
            task: str,
            context: dict[str, Any],
            max_attempts: int = 3,
        ) -> dict[str, Any]:
            """Capture the existing persistent-search result unchanged."""

            cached_row = {
                "memory_name": shared_memory_manifest["memory_name"],
                "content": shared_memory_manifest["content"],
                "timestamp": "2026-07-14T00:00:00Z",
                "source_global_user_id": _CHARACTER_GLOBAL_ID,
                "memory_type": "fact",
                "source_kind": "seeded_manual",
                "status": "active",
                "methods": ["test_seeded_shared_memory"],
                "matched_anchors": [],
                "score": 1.0,
                "hybrid_rank": 1,
            }
            cache_key = build_persistent_memory_search_cache_key(
                task,
                context,
            )
            await worker.write_cache(  # type: ignore[attr-defined]
                cache_key=cache_key,
                result=[cached_row],
                dependencies=build_persistent_memory_search_dependencies({}),
                metadata={"test_fixture": "shared_character_or_world"},
            )
            result = await original_worker_run(  # type: ignore[arg-type]
                worker,
                task,
                context,
                max_attempts,
            )
            worker_results.append(_json_safe(result))
            return result

        monkeypatch.setattr(
            capabilities.PersistentMemorySearchAgent,
            "run",
            capture_worker_result,
        )
        original_generator = persistent_search._generator
        original_tool = persistent_search._tool
        original_judge = persistent_search._judge

        async def capture_generator(
            task: str,
            context: dict[str, Any],
            feedback: str,
        ) -> dict[str, Any]:
            """Capture generated search arguments without changing them."""

            result = await original_generator(task, context, feedback)
            worker_diagnostics.append({
                "kind": "generator",
                "task": task,
                "result": _json_safe(result),
            })
            return result

        async def capture_tool(args: dict[str, Any]) -> object:
            """Capture persistent-search rows without changing them."""

            result = await original_tool(args)
            worker_diagnostics.append({
                "kind": "tool",
                "args": _json_safe(args),
                "result": _json_safe(result),
            })
            return result

        async def capture_judge(task: str, result: object) -> tuple[bool, str]:
            """Capture the worker judge verdict without changing it."""

            verdict = await original_judge(task, result)
            worker_diagnostics.append({
                "kind": "judge",
                "result": _json_safe(result),
                "verdict": _json_safe(verdict),
            })
            return verdict

        monkeypatch.setattr(persistent_search, "_generator", capture_generator)
        monkeypatch.setattr(persistent_search, "_tool", capture_tool)
        monkeypatch.setattr(persistent_search, "_judge", capture_judge)
        started_at = perf_counter()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.app),
            base_url="http://relational-willingness-e2e",
            timeout=None,
        ) as client:
            health_response = await client.get("/health")
            if health_response.status_code != 200:
                raise AssertionError(
                    f"guarded service health failed: {health_response.status_code}"
                )
            original_trace_assertion = (
                replay_harness._assert_full_trace_capture
            )

            def require_relational_trace_capture(trace_steps: object) -> None:
                """Apply strict raw-model capture with typed exceptions."""

                _assert_relational_trace_capture(
                    trace_steps,
                    original_trace_assertion,
                )

            replay_harness._assert_full_trace_capture = (
                require_relational_trace_capture
            )
            try:
                with _capture_raw_llm_steps(raw_llm_calls):
                    turn_artifact = await _run_one_turn(
                        client=client,
                        database=database,
                        adapter=adapter,
                        request=request,
                        source_message=source_message,
                        prior_turn=None,
                    )
            finally:
                replay_harness._assert_full_trace_capture = (
                    original_trace_assertion
                )
        response = turn_artifact["response"]
        if not isinstance(response, Mapping):
            raise AssertionError("public ChatResponse evidence is not an object")
        trace_steps = turn_artifact.get("trace_steps", [])
        decision = _find_relational_decision(trace_steps)
        if decision is None:
            raise AssertionError("protected trace has no relational willingness decision")
        cognition_graph = response.get("cognition_graph")
        collapse_node, branch_nodes = _cognition_graph_collapse(
            cognition_graph,
        )
        collapse_detail = collapse_node.get("detail")
        if not isinstance(collapse_detail, Mapping):
            raise AssertionError("public V2 collapse detail is missing")
        collapse_observation = collapse_detail.get("collapse")
        if not isinstance(collapse_observation, Mapping):
            raise AssertionError(
                "public V2 collapse observation is missing"
            )
        if collapse_node.get("status") != "completed":
            raise AssertionError(
                f"public V2 collapse did not complete: {collapse_node}"
            )
        if decision.get("applicability") == "relationship_sensitive":
            if collapse_observation.get("selection_reason") != (
                facade_module.AUTHORITATIVE_RELATIONAL_COLLAPSE_REASON
            ):
                raise AssertionError(
                    "sensitive relational collapse lacks fixed preservation reason"
                )
            if collapse_observation.get("supporting_branch_indices", []) != []:
                raise AssertionError(
                    "sensitive relational collapse exposed supporting branches: "
                    f"decision={decision} collapse={collapse_observation} "
                    f"branches={branch_nodes}"
                )
            primary_index = collapse_observation.get("primary_branch_index")
            primary_branch = next(
                (
                    branch
                    for branch in branch_nodes
                    if isinstance(branch.get("detail"), Mapping)
                    and branch["detail"].get("branch_index") == primary_index
                ),
                None,
            )
            if not isinstance(primary_branch, Mapping):
                raise AssertionError(
                    "sensitive relational collapse has no primary branch"
                )
            primary_detail = primary_branch.get("detail")
            if not isinstance(primary_detail, Mapping) or (
                primary_detail.get("goal_kind") != "ordinary_response"
                or primary_detail.get("selection") != "primary"
            ):
                raise AssertionError(
                    "sensitive relational collapse did not preserve ordinary owner"
                )
        stage_names = _trace_stage_names(trace_steps)
        trace_capture_gaps = _trace_capture_gaps(trace_steps)
        stage_text = " ".join(stage_names).casefold()
        if "goal" not in stage_text:
            raise AssertionError(f"ordinary goal stage missing: {stage_names}")
        if "action" not in stage_text:
            raise AssertionError(f"action stage missing: {stage_names}")
        if not any(
            marker in stage_text
            for marker in ("surface", "dialog", "content_plan")
        ):
            raise AssertionError(f"L3 surface stage missing: {stage_names}")
        critical_stage_markers = ("goal", "action", "dialog")
        critical_gaps = [
            stage_name
            for stage_name in trace_capture_gaps
            if any(marker in stage_name.casefold() for marker in critical_stage_markers)
        ]
        if critical_gaps:
            raise AssertionError(
                f"critical public path trace capture gaps: {critical_gaps}"
            )
        response_surface = turn_artifact.get("response_surface")
        if not isinstance(response_surface, Mapping):
            raise AssertionError("public response surface evidence is missing")
        if response_surface.get("status") != "visible_dialog":
            raise AssertionError(
                f"expected visible dialog, got {response_surface}"
            )
        visible_text = _visible_text(response)
        if not visible_text:
            raise AssertionError("public visible dialog is empty")
        actual_stance = str(decision.get("stance", ""))
        if actual_stance not in _VALID_STANCES:
            raise AssertionError(f"invalid public stance: {actual_stance}")
        decision_state = str(
            decision.get("current_user_relationship_state", "")
        )
        if decision_state not in {
            "not_applicable",
            "unestablished",
            "developing_or_uncertain",
            "established",
        }:
            raise AssertionError(
                f"invalid public relationship state: {decision_state}"
            )
        if actual_stance != "not_applicable" and (
            decision.get("applicability") != "relationship_sensitive"
        ):
            raise AssertionError(
                "non-applicable stance requires a sensitive applicability: "
                f"decision={decision}"
            )
        if profile_name == "stranger" and decision_state != "unestablished":
            raise AssertionError(
                f"stranger expected unestablished state, got {decision_state}; "
                f"decision={decision}"
            )
        if profile_name == "lover" and decision_state != "established":
            raise AssertionError(
                f"lover expected established state, got {decision_state}; "
                f"decision={decision}"
            )
        endpoint_expectations = fixture["endpoint_expectations"]
        expected_stance = str(endpoint_expectations.get(profile_name, ""))
        if expected_stance and expected_stance != "observation_only":
            if actual_stance != expected_stance:
                raise AssertionError(
                    f"{profile_name} expected {expected_stance}, "
                    f"got {actual_stance}"
                )
        raw_prompt_text = json.dumps(raw_llm_calls, ensure_ascii=False)
        shared_memory_in_prompt = (
            shared_memory_manifest["content"] in raw_prompt_text
        )
        if not shared_memory_in_prompt:
            memory_diagnostics = []
            for call in raw_llm_calls:
                serialized_call = json.dumps(call, ensure_ascii=False)
                if not any(
                    marker in serialized_call.casefold()
                    for marker in ("memory", "rag", "persistent")
                ):
                    continue
                memory_diagnostics.append({
                    "stage_name": call.get("stage_name", ""),
                    "raw_response_text": str(
                        call.get("raw_response_text", "")
                    )[:500],
                    "parsed_output": call.get("parsed_output", {}),
                })
            raise AssertionError(
                "seeded shared-memory evidence did not reach a captured prompt; "
                f"stages={stage_names}; diagnostics={memory_diagnostics}; "
                f"prewarm_results={prewarm_results}; "
                f"worker_results={worker_results}; "
                f"worker_diagnostics={worker_diagnostics}"
            )
        raw_id_leak_locations: list[dict[str, str]] = []
        if global_user_id in raw_prompt_text:
            for call in raw_llm_calls:
                stage_name = str(call.get("stage_name", ""))
                raw_messages = call.get("raw_messages")
                if not isinstance(raw_messages, list):
                    continue
                for message in raw_messages:
                    if not isinstance(message, Mapping):
                        continue
                    content = str(message.get("content", ""))
                    if global_user_id not in content:
                        continue
                    raw_id_leak_locations.append({
                        "stage_name": stage_name,
                        "role": str(message.get("role", "")),
                    })
            core_id_leaks = [
                location
                for location in raw_id_leak_locations
                if location["stage_name"] != "message_decontextualizer"
            ]
            if core_id_leaks:
                raise AssertionError(
                    "raw live global user id leaked into a core prompt: "
                    f"locations={core_id_leaks}"
                )
        artifact = {
            "schema_version": (
                "cognition_core_v2_relational_willingness_e2e.v1"
            ),
            "profile_name": profile_name,
            "sample_index": sample_index,
            "database_name": database.name,
            "database_guard": os.environ.get("KAZUSA_TEST_DB_GUARD", ""),
            "guarded_runtime": guarded_runtime,
            "fixture_path": str(_FIXTURE_PATH),
            "request": fixture["request"],
            "relationship_profile": relationship_profile,
            "hash_manifest": hash_manifest,
            "scene": fixture["scene"],
            "shared_memory": shared_memory_manifest,
            "shared_memory_in_prompt": shared_memory_in_prompt,
            "prewarm_results": prewarm_results,
            "worker_results": worker_results,
            "worker_diagnostics": worker_diagnostics,
            "character_identity_sha256": hashlib.sha256(
                _CHARACTER_PATH.read_bytes()
            ).hexdigest(),
            "user_global_id": global_user_id,
            "response": _json_safe(response),
            "visible_text": visible_text,
            "response_surface": _json_safe(response_surface),
            "cognition_graph": _json_safe(cognition_graph),
            "collapse_node": _json_safe(collapse_node),
            "decision": _json_safe(decision),
            "trace_stage_names": stage_names,
            "trace_capture_gaps": trace_capture_gaps,
            "turn_artifact": _json_safe(turn_artifact),
            "raw_llm_calls": _json_safe(raw_llm_calls),
            "raw_current_user_id_leak_locations": raw_id_leak_locations,
            "adapter_calls": _json_safe(adapter.calls),
            "duration_ms": round((perf_counter() - started_at) * 1000),
        }
        artifact_path = _write_artifact(profile_name, artifact)
        print(json.dumps({
            "profile_name": profile_name,
            "stance": actual_stance,
            "visible_text": visible_text,
            "artifact": str(artifact_path),
        }, ensure_ascii=False))
        return artifact


@pytest.mark.parametrize("sample_index", (1, 2, 3))
async def test_stranger_visible_rejection(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    sample_index: int,
) -> None:
    """Three fresh stranger samples visibly refuse through ``/chat``."""

    await _run_public_case(
        live_db=live_db,
        monkeypatch=monkeypatch,
        profile_name="stranger",
        sample_index=sample_index,
    )


@pytest.mark.parametrize("sample_index", (1, 2, 3))
async def test_lover_visible_acceptance(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    sample_index: int,
) -> None:
    """Three fresh lover samples visibly accept through ``/chat``."""

    await _run_public_case(
        live_db=live_db,
        monkeypatch=monkeypatch,
        profile_name="lover",
        sample_index=sample_index,
    )


@pytest.mark.parametrize("sample_index", (1, 2, 3))
async def test_intermediate_33_visible_observation(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    sample_index: int,
) -> None:
    """Retain three visible one-third relationship observations."""

    await _run_public_case(
        live_db=live_db,
        monkeypatch=monkeypatch,
        profile_name="intermediate_33",
        sample_index=sample_index,
    )


@pytest.mark.parametrize("sample_index", (1, 2, 3))
async def test_intermediate_67_visible_observation(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    sample_index: int,
) -> None:
    """Retain three visible two-thirds relationship observations."""

    await _run_public_case(
        live_db=live_db,
        monkeypatch=monkeypatch,
        profile_name="intermediate_67",
        sample_index=sample_index,
    )
