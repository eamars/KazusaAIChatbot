from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import kazusa_ai_chatbot.llm_tracing as tracing
from kazusa_ai_chatbot import utils as utils_module
from kazusa_ai_chatbot.llm_tracing import failure_capsule

from llm_test_helpers import make_llm_call_config


@pytest.mark.asyncio
async def test_record_trace_step_metadata_mode_omits_raw_payload(monkeypatch):
    written: list[dict] = []

    async def insert_step(document: dict) -> str:
        written.append(document)
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    result = await tracing.record_llm_trace_step(
        trace_id="trace-1",
        stage_name="dialog_generator",
        route_name="DIALOG_GENERATOR_LLM",
        model_name="model-a",
        messages=[
            SystemMessage(content="system secret"),
            HumanMessage(content="hello"),
        ],
        response_text='{"final_dialog":["hi"]}',
        parsed_output={"final_dialog": ["hi"]},
        parse_status="succeeded",
        status="succeeded",
        duration_ms=10,
        output_state_fields=["final_dialog"],
    )

    assert result["status"] == "recorded"
    assert len(written) == 1
    doc = written[0]
    assert doc["prompt_chars"] == len("system secret") + len("hello")
    assert doc["output_chars"] == len('{"final_dialog":["hi"]}')
    assert doc["prompt_sha256"]
    assert doc["output_sha256"]
    assert doc["raw_messages"] == []
    assert doc["raw_response_text"] == ""
    assert doc["parsed_output"] == {}
    assert isinstance(doc["expires_at"], datetime)


@pytest.mark.asyncio
async def test_record_trace_step_off_mode_skips_db_write(monkeypatch):
    insert_step = AsyncMock()

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "off")
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    result = await tracing.record_llm_trace_step(
        trace_id="trace-1",
        stage_name="stage",
        route_name="route",
        model_name="model",
        messages=[HumanMessage(content="hello")],
        response_text="{}",
        parsed_output={},
        parse_status="succeeded",
        status="succeeded",
        duration_ms=1,
        output_state_fields=[],
    )

    assert result["status"] == "skipped"
    insert_step.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_trace_run_updates_status(monkeypatch):
    update_run = AsyncMock()

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(tracing.db_llm_tracing, "update_trace_run", update_run)

    await tracing.finalize_llm_trace_run(
        trace_id="trace-1",
        status="succeeded",
        final_dialog_count=1,
        delivery_tracking_id="delivery-1",
    )

    update_doc = update_run.await_args.kwargs["update_doc"]
    assert update_doc["status"] == "succeeded"
    assert update_doc["final_dialog_count"] == 1
    assert update_doc["delivery_tracking_id"] == "delivery-1"


def test_build_trace_id_is_prefixed():
    trace_id = tracing.build_trace_id()

    assert trace_id.startswith("llmtrace_")


@pytest.mark.asyncio
async def test_failure_capsule_promotes_exact_input_and_attempt(monkeypatch):
    written: list[dict] = []
    persisted = asyncio.Event()

    async def insert_step(document: dict) -> str:
        written.append(document)
        if document.get("capture_reason") == "cognition_failure_capsule":
            persisted.set()
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    input_payload = {
        "schema_version": "cognition_core_input.v2",
        "nested": {"value": "before"},
    }
    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-capsule",
        entrypoint="run_cognition",
        input_payload=input_payload,
    )
    config = make_llm_call_config("cognition_test_stage")
    await tracing.record_llm_trace_step(
        trace_id="trace-capsule",
        stage_name="cognition_test_stage",
        route_name=config.route_name,
        model_name=config.model,
        messages=[
            SystemMessage(content="exact system prompt"),
            HumanMessage(content="exact user payload"),
        ],
        response_text='{"selection":"invalid"}',
        parsed_output={"selection": "invalid"},
        parse_status="contract_error",
        status="recovered",
        duration_ms=10,
        output_state_fields=[],
        call_config=config,
        branch_id="autonomy_boundary",
        attempt_index=1,
        validation_error="required field missing",
    )
    failure_capsule.mark_failure(
        session,
        failure_kind="recovered_contract_error",
        stage_name="cognition_test_stage",
        details={"branch_id": "autonomy_boundary"},
    )
    input_payload["nested"]["value"] = "after"

    invocation_id = failure_capsule.finish_failure_capsule(
        session,
        outcome="partial_failure",
    )
    await asyncio.wait_for(persisted.wait(), timeout=1)

    capsule_rows = [
        row
        for row in written
        if row.get("capture_reason") == "cognition_failure_capsule"
    ]
    assert len(capsule_rows) == 1
    capsule = capsule_rows[0]["capsule"]
    assert capsule["cognition_invocation_id"] == invocation_id
    assert capsule["input_payload"]["nested"]["value"] == "before"
    assert capsule["input_sha256"]
    assert capsule["outcome"] == "partial_failure"
    assert capsule["failure_events"] == [{
        "failure_kind": "recovered_contract_error",
        "stage_name": "cognition_test_stage",
        "details": {"branch_id": "autonomy_boundary"},
    }]
    assert capsule["attempts"][0]["messages"] == [
        {"role": "system", "content": "exact system prompt"},
        {"role": "human", "content": "exact user payload"},
    ]
    assert capsule["attempts"][0]["raw_response_text"] == (
        '{"selection":"invalid"}'
    )
    assert capsule["attempts"][0]["validation_error"] == (
        "required field missing"
    )
    assert capsule["attempts"][0]["branch_id"] == "autonomy_boundary"
    assert capsule["attempts"][0]["config"]["base_url"] == config.base_url
    assert "api_key" not in capsule["attempts"][0]["config"]
    assert "test-api-key" not in repr(capsule_rows[0])


@pytest.mark.asyncio
async def test_json_repair_attempt_uses_explicit_capsule_hook(monkeypatch):
    written: list[dict] = []
    persisted = asyncio.Event()

    async def insert_step(document: dict) -> str:
        written.append(document)
        persisted.set()
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)
    monkeypatch.setattr(
        utils_module._parse_json_with_llm,
        "invoke",
        lambda messages, *, config: SimpleNamespace(
            content='{"repaired":true}'
        ),
    )
    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-json-repair",
        entrypoint="run_cognition",
        input_payload={"value": "broken"},
    )

    parsed = utils_module.parse_json_with_llm(
        "{repaired: true}",
        repair_trace_hook=failure_capsule.append_json_repair_attempt,
    )
    invocation_id = failure_capsule.finish_failure_capsule(
        session,
        outcome=None,
    )
    await asyncio.wait_for(persisted.wait(), timeout=1)

    assert parsed == {"repaired": True}
    assert invocation_id
    capsule = written[0]["capsule"]
    assert capsule["failure_events"] == [{
        "failure_kind": "recovered_json_repair",
        "stage_name": "json_repair",
        "details": {},
    }]
    attempt = capsule["attempts"][0]
    assert attempt["stage_name"] == "json_repair"
    assert attempt["raw_response_text"] == '{"repaired":true}'
    assert attempt["parse_status"] == "succeeded"
    assert "api_key" not in attempt["config"]


@pytest.mark.asyncio
async def test_clean_failure_capsule_session_discards_without_write(monkeypatch):
    written: list[dict] = []

    async def insert_step(document: dict) -> str:
        written.append(document)
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-clean",
        entrypoint="run_cognition",
        input_payload={"value": "clean"},
    )
    await tracing.record_llm_trace_step(
        trace_id="trace-clean",
        stage_name="clean-stage",
        route_name="TEST_LLM",
        model_name="test-model",
        messages=[HumanMessage(content="clean prompt")],
        response_text='{"value":"clean"}',
        parsed_output={"value": "clean"},
        parse_status="succeeded",
        status="succeeded",
        duration_ms=1,
        output_state_fields=[],
        branch_id="ordinary_response",
        attempt_index=1,
        validation_error="",
    )
    result = failure_capsule.finish_failure_capsule(
        session,
        outcome=None,
    )
    await asyncio.sleep(0)

    assert result == ""
    assert len(written) == 1
    assert "capture_reason" not in written[0]
    assert written[0]["raw_messages"] == []
    assert written[0]["raw_response_text"] == ""


def test_failure_capsule_snapshot_error_is_contained(monkeypatch, caplog):
    class Uncopyable:
        """Raise when protected capture tries to snapshot the value."""

        def __deepcopy__(self, memo):
            """Simulate an input snapshot failure."""

            del memo
            raise RuntimeError("protected input detail")

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )

    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-snapshot-error",
        entrypoint="run_cognition",
        input_payload={"value": Uncopyable()},
    )

    assert session is None
    assert "protected input detail" not in caplog.text


@pytest.mark.asyncio
async def test_capsule_persistence_never_blocks_caller(monkeypatch):
    persistence_started = asyncio.Event()
    release_persistence = asyncio.Event()

    async def insert_step(document: dict) -> str:
        persistence_started.set()
        await release_persistence.wait()
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-blocked",
        entrypoint="run_cognition",
        input_payload={"value": "failed"},
    )
    failure_capsule.mark_failure(
        session,
        failure_kind="terminal_failure",
        stage_name="run_cognition",
        details={},
    )

    invocation_id = failure_capsule.finish_failure_capsule(
        session,
        outcome="terminal_failure",
        exception=RuntimeError("original failure"),
    )

    assert invocation_id
    await asyncio.wait_for(persistence_started.wait(), timeout=1)
    release_persistence.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_capsule_stalled_persistence_is_bounded_and_released(monkeypatch):
    persistence_started = asyncio.Event()
    release_persistence = asyncio.Event()
    pending_before = set(failure_capsule._PENDING_PERSISTENCE_TASKS)

    async def insert_step(document: dict) -> str:
        persistence_started.set()
        await release_persistence.wait()
        return document["step_id"]

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        failure_capsule,
        "FAILURE_CAPSULE_WRITE_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-stalled",
        entrypoint="run_cognition",
        input_payload={"value": "failed"},
    )
    invocation_id = failure_capsule.finish_failure_capsule(
        session,
        outcome="terminal_failure",
        exception=RuntimeError("original failure"),
    )

    assert invocation_id
    await asyncio.wait_for(persistence_started.wait(), timeout=1)
    for _ in range(20):
        if set(failure_capsule._PENDING_PERSISTENCE_TASKS) == pending_before:
            break
        await asyncio.sleep(0.01)

    assert set(failure_capsule._PENDING_PERSISTENCE_TASKS) == pending_before


@pytest.mark.asyncio
async def test_capsule_persistence_failure_is_sanitized(monkeypatch, caplog):
    persistence_attempted = asyncio.Event()

    async def insert_step(document: dict) -> str:
        del document
        persistence_attempted.set()
        raise RuntimeError("raw input and api key must stay protected")

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)
    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-write-failure",
        entrypoint="run_cognition",
        input_payload={"secret_input": "protected value"},
    )
    failure_capsule.mark_failure(
        session,
        failure_kind="failed_branch",
        stage_name="goal_cognition",
        details={},
    )

    invocation_id = failure_capsule.finish_failure_capsule(
        session,
        outcome="partial_failure",
    )
    await asyncio.wait_for(persistence_attempted.wait(), timeout=1)
    await asyncio.sleep(0)

    assert invocation_id
    assert "protected value" not in caplog.text
    assert "raw input and api key" not in caplog.text


@pytest.mark.asyncio
async def test_capsule_task_scheduling_failure_is_contained(
    monkeypatch,
    caplog,
):
    def reject_task(coroutine):
        coroutine.close()
        raise RuntimeError("protected scheduling detail")

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(failure_capsule.asyncio, "create_task", reject_task)
    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-scheduling-failure",
        entrypoint="run_cognition",
        input_payload={"secret_input": "protected value"},
    )
    failure_capsule.mark_failure(
        session,
        failure_kind="terminal_failure",
        stage_name="run_cognition",
        details={},
    )

    invocation_id = failure_capsule.finish_failure_capsule(
        session,
        outcome="terminal_failure",
        exception=RuntimeError("original cognition error"),
    )

    assert invocation_id
    assert "protected scheduling detail" not in caplog.text
    assert "protected value" not in caplog.text


@pytest.mark.asyncio
async def test_failure_capsule_context_isolated_between_concurrent_calls(
    monkeypatch,
):
    written: list[dict] = []
    both_started = asyncio.Event()
    started_count = 0

    async def insert_step(document: dict) -> str:
        written.append(document)
        return document["step_id"]

    async def run_capture(value: str) -> str:
        nonlocal started_count
        session = failure_capsule.begin_failure_capsule(
            trace_id="trace-concurrent",
            entrypoint="run_cognition",
            input_payload={"value": value},
        )
        started_count += 1
        if started_count == 2:
            both_started.set()
        await both_started.wait()
        await tracing.record_llm_trace_step(
            trace_id="trace-concurrent",
            stage_name=f"stage-{value}",
            route_name="TEST_LLM",
            model_name="test-model",
            messages=[HumanMessage(content=f"prompt-{value}")],
            response_text=f"response-{value}",
            parsed_output={"value": value},
            parse_status="contract_error",
            status="failed",
            duration_ms=1,
            output_state_fields=[],
            branch_id=value,
            attempt_index=1,
            validation_error=f"error-{value}",
        )
        failure_capsule.mark_failure(
            session,
            failure_kind="failed_branch",
            stage_name=f"stage-{value}",
            details={"branch_id": value},
        )
        invocation_id = failure_capsule.finish_failure_capsule(
            session,
            outcome="partial_failure",
        )
        return invocation_id

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "metadata")
    monkeypatch.setattr(
        failure_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    invocation_ids = await asyncio.gather(
        run_capture("first"),
        run_capture("second"),
    )
    for _ in range(10):
        capsule_rows = [
            row
            for row in written
            if row.get("capture_reason") == "cognition_failure_capsule"
        ]
        if len(capsule_rows) == 2:
            break
        await asyncio.sleep(0)

    assert len(set(invocation_ids)) == 2
    capsules = {
        row["capsule"]["input_payload"]["value"]: row["capsule"]
        for row in capsule_rows
    }
    assert capsules["first"]["attempts"][0]["branch_id"] == "first"
    assert capsules["first"]["attempts"][0]["raw_response_text"] == (
        "response-first"
    )
    assert capsules["second"]["attempts"][0]["branch_id"] == "second"
    assert capsules["second"]["attempts"][0]["raw_response_text"] == (
        "response-second"
    )


@pytest.mark.asyncio
async def test_failure_capsule_off_mode_stores_nothing(monkeypatch):
    insert_step = AsyncMock()
    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "off")
    monkeypatch.setattr(failure_capsule, "LLM_TRACE_CAPTURE_MODE", "off")
    monkeypatch.setattr(tracing.db_llm_tracing, "insert_trace_step", insert_step)

    session = failure_capsule.begin_failure_capsule(
        trace_id="trace-off",
        entrypoint="run_cognition",
        input_payload={"value": "failed"},
    )
    result = failure_capsule.finish_failure_capsule(
        session,
        outcome="terminal_failure",
        exception=RuntimeError("failure"),
    )
    await asyncio.sleep(0)

    assert session is None
    assert result == ""
    insert_step.assert_not_awaited()
