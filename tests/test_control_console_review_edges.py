"""Focused review-edge tests for control-console contracts."""

from __future__ import annotations

import pytest


def test_service_config_descriptor_and_registry_reject_bad_shapes() -> None:
    """Invalid descriptors should fail before operators can see them."""

    from pydantic import ValidationError

    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigRegistry,
        ServiceConfigValidationError,
    )

    with pytest.raises(ValidationError, match="invalid field pattern"):
        ServiceConfigField(
            key="bad_pattern",
            label="Bad pattern",
            description="Bad pattern descriptor.",
            value_type="string",
            pattern="[",
        )

    duplicate_field = ServiceConfigField(
        key="enabled",
        label="Enabled",
        description="Whether the adapter participates.",
        value_type="boolean",
    )
    with pytest.raises(ValidationError, match="duplicate config field"):
        ServiceConfigDescriptor(
            service_id="adapter.fake",
            title="Fake adapter",
            description="Duplicate field descriptor.",
            fields=[duplicate_field, duplicate_field],
        )

    descriptor = ServiceConfigDescriptor(
        service_id="adapter.fake",
        title="Fake adapter",
        description="Duplicate service descriptor.",
        fields=[duplicate_field],
    )
    with pytest.raises(ServiceConfigValidationError, match="duplicate service"):
        ServiceConfigRegistry(descriptors=[descriptor, descriptor])


def test_service_config_generic_defaults_validation_and_rendering() -> None:
    """Generic descriptors should validate defaults, overrides, and argv output."""

    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigOverrideStore,
        ServiceConfigRegistry,
        ServiceConfigValidationError,
    )

    descriptor = ServiceConfigDescriptor(
        service_id="adapter.generic",
        title="Generic adapter",
        description="Generic adapter descriptor.",
        fields=[
            ServiceConfigField(
                key="enabled",
                label="Enabled",
                description="Whether the adapter participates.",
                value_type="boolean",
                default_env="GENERIC_ENABLED",
            ),
            ServiceConfigField(
                key="max_turns",
                label="Max turns",
                description="Maximum turn count.",
                value_type="integer",
                default_env="GENERIC_MAX_TURNS",
            ),
            ServiceConfigField(
                key="mode",
                label="Mode",
                description="Adapter mode.",
                value_type="enum",
                default_env="GENERIC_MODE",
                options=["safe", "live"],
            ),
            ServiceConfigField(
                key="token",
                label="Token",
                description="Sensitive token.",
                value_type="string",
                default_env="GENERIC_TOKEN",
                sensitive=True,
            ),
        ],
    )

    def renderer(
        base_command: list[str],
        effective_values: dict[str, object],
    ) -> list[str]:
        mode = effective_values["mode"]
        max_turns = effective_values["max_turns"]
        rendered_command = [
            *base_command,
            "--mode",
            str(mode),
            "--max-turns",
            str(max_turns),
        ]
        return rendered_command

    registry = ServiceConfigRegistry(
        descriptors=[descriptor],
        command_renderers={"adapter.generic": renderer},
    )
    overrides = ServiceConfigOverrideStore()
    environment = {
        "GENERIC_ENABLED": "yes",
        "GENERIC_MAX_TURNS": "7",
        "GENERIC_MODE": "safe",
        "GENERIC_TOKEN": "secret",
    }

    snapshot = registry.snapshot_for_service(
        service_id="adapter.generic",
        environment=environment,
        overrides=overrides,
    )
    fields_by_key = {field.key: field for field in snapshot.fields}

    assert fields_by_key["enabled"].effective_value is True
    assert fields_by_key["max_turns"].effective_value == 7
    assert fields_by_key["mode"].validation["options"] == ["safe", "live"]
    assert fields_by_key["token"].default_value is None
    assert fields_by_key["token"].effective_value is None

    overrides.set_override(
        service_id="adapter.generic",
        values={"enabled": False, "max_turns": 3, "mode": "live"},
        registry=registry,
        environment=environment,
    )
    rendered = registry.render_start_command(
        service_id="adapter.generic",
        base_command=["python", "-m", "adapter.generic"],
        environment=environment,
        overrides=overrides,
    )

    assert rendered == [
        "python",
        "-m",
        "adapter.generic",
        "--mode",
        "live",
        "--max-turns",
        "3",
    ]

    invalid_values = [
        {"unknown": "value"},
        {"enabled": "false"},
        {"max_turns": True},
        {"max_turns": "3"},
        {"mode": "unsupported"},
        {"token": ""},
    ]
    for values in invalid_values:
        with pytest.raises(ServiceConfigValidationError):
            registry.validate_values(
                service_id="adapter.generic",
                values=values,
                environment=environment,
            )

    with pytest.raises(ServiceConfigValidationError, match="boolean"):
        registry.snapshot_for_service(
            service_id="adapter.generic",
            environment={**environment, "GENERIC_ENABLED": "maybe"},
            overrides=ServiceConfigOverrideStore(),
        )
    with pytest.raises(ServiceConfigValidationError, match="integer"):
        registry.snapshot_for_service(
            service_id="adapter.generic",
            environment={**environment, "GENERIC_MAX_TURNS": "3.5"},
            overrides=ServiceConfigOverrideStore(),
        )


def test_service_config_rejects_invalid_renderer_output() -> None:
    """Command renderers must return bounded argv lists, not shell strings."""

    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigOverrideStore,
        ServiceConfigRegistry,
        ServiceConfigValidationError,
    )

    descriptor = ServiceConfigDescriptor(
        service_id="adapter.bad",
        title="Bad adapter",
        description="Bad renderer descriptor.",
        fields=[
            ServiceConfigField(
                key="mode",
                label="Mode",
                description="Adapter mode.",
                value_type="string",
                default_env="BAD_MODE",
            ),
        ],
    )

    def empty_renderer(
        base_command: list[str],
        effective_values: dict[str, object],
    ) -> list[str]:
        _ = base_command
        _ = effective_values
        return []

    registry = ServiceConfigRegistry(
        descriptors=[descriptor],
        command_renderers={"adapter.bad": empty_renderer},
    )

    with pytest.raises(ServiceConfigValidationError, match="non-empty list"):
        registry.render_start_command(
            service_id="adapter.bad",
            base_command=["python", "-m", "adapter.bad"],
            environment={"BAD_MODE": "live"},
            overrides=ServiceConfigOverrideStore(),
        )

    def blank_part_renderer(
        base_command: list[str],
        effective_values: dict[str, object],
    ) -> list[str]:
        _ = effective_values
        return [*base_command, ""]

    blank_registry = ServiceConfigRegistry(
        descriptors=[descriptor],
        command_renderers={"adapter.bad": blank_part_renderer},
    )

    with pytest.raises(ServiceConfigValidationError, match="non-empty strings"):
        blank_registry.render_start_command(
            service_id="adapter.bad",
            base_command=["python", "-m", "adapter.bad"],
            environment={"BAD_MODE": "live"},
            overrides=ServiceConfigOverrideStore(),
        )


def test_service_config_remaining_validation_edges() -> None:
    """Remaining validation branches should reject unsafe operator input."""

    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigOverrideStore,
        ServiceConfigRegistry,
        ServiceConfigValidationError,
        _render_napcat_command,
    )

    descriptor = ServiceConfigDescriptor(
        service_id="adapter.edges",
        title="Edge adapter",
        description="Edge validation descriptor.",
        fields=[
            ServiceConfigField(
                key="groups",
                label="Groups",
                description="Group list.",
                value_type="string_list",
                pattern=r"^[0-9]+$",
            ),
            ServiceConfigField(
                key="name",
                label="Name",
                description="Display name.",
                value_type="string",
                default_env="EDGE_NAME",
                max_item_length=4,
            ),
            ServiceConfigField(
                key="enabled",
                label="Enabled",
                description="Whether this service participates.",
                value_type="boolean",
                default_env="EDGE_ENABLED",
            ),
            ServiceConfigField(
                key="count",
                label="Count",
                description="Count value.",
                value_type="integer",
                default_env="EDGE_COUNT",
            ),
        ],
    )
    registry = ServiceConfigRegistry(descriptors=[descriptor])
    overrides = ServiceConfigOverrideStore()
    default_environment = {"EDGE_NAME": "ok"}

    default_snapshot = registry.snapshot_for_service(
        service_id="adapter.edges",
        environment=default_environment,
        overrides=overrides,
    )
    defaults = {field.key: field.effective_value for field in default_snapshot.fields}

    assert defaults["groups"] == []
    assert defaults["enabled"] is False
    assert defaults["count"] == 0
    assert registry.render_start_command(
        service_id="adapter.edges",
        base_command=["python", "-m", "adapter.edges"],
        environment=default_environment,
        overrides=overrides,
    ) == ["python", "-m", "adapter.edges"]

    invalid_values = [
        {"groups": "123"},
        {"groups": [123]},
        {"groups": ["abc"]},
        {"name": 123},
        {"name": "abcde"},
    ]
    for values in invalid_values:
        with pytest.raises(ServiceConfigValidationError):
            registry.validate_values(
                service_id="adapter.edges",
                values=values,
                environment=default_environment,
            )

    with pytest.raises(ServiceConfigValidationError, match="no config descriptor"):
        registry.validate_values(
            service_id="adapter.missing",
            values={"groups": []},
            environment={},
        )
    with pytest.raises(ServiceConfigValidationError, match="active_groups"):
        _render_napcat_command(
            ["python", "-m", "adapters.napcat_qq_adapter"],
            {"active_groups": "bad"},
        )


@pytest.mark.asyncio
async def test_repository_owner_pages_write_off_unsupported_surfaces() -> None:
    """Owner pages should be explicit when the current backend lacks a surface."""

    from control_console.repository import ControlConsoleRepository

    async def missing_user_profile(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> None:
        assert identifier == "missing-user"
        assert platform == "qq"
        return None

    async def group_style_context(
        *,
        global_user_id: str,
        channel_type: str,
        platform: str,
        platform_channel_id: str,
    ) -> dict[str, object]:
        assert global_user_id == ""
        assert channel_type == "group"
        assert platform == "qq"
        assert platform_channel_id == "group-1"
        return {"application_order": []}

    repository = ControlConsoleRepository(
        find_user_profile_by_identifier=missing_user_profile,
        build_interaction_style_context=group_style_context,
    )

    user = await repository.lookup_user_entity(
        platform="qq",
        platform_user_id="missing-user",
        query="",
        limit=5,
    )
    missing_group_platform = await repository.lookup_group_entity(
        platform="",
        group_id="group-1",
        limit=5,
    )
    group = await repository.lookup_group_entity(
        platform="qq",
        group_id="group-1",
        limit=5,
    )

    assert user["status"] == "empty"
    assert user["identity"]["resolution_status"] == "empty"
    assert set(user["panels"]) == {
        "profile",
        "relationship",
        "memory",
        "style",
        "conversation_progress_prompt",
        "current_carry_over",
    }
    assert all(panel["items"] == [] for panel in user["panels"].values())
    assert missing_group_platform["status"] == "needs_input"
    assert "platform is required" in missing_group_platform["panels"]["style"]["reason"]
    assert group["status"] == "needs_input"
    assert group["panels"]["style"]["reason"] == (
        "no interaction-style guidance matched the lookup"
    )
    assert "not exposed" in group["panels"]["progress"]["reason"]
    assert "not available" in group["panels"]["guidance"]["reason"]


@pytest.mark.asyncio
async def test_repository_character_projection_fallbacks_are_readable() -> None:
    """Character projection should keep readable fields and safe fallbacks."""

    from control_console.repository import ControlConsoleRepository

    async def get_character_profile() -> dict[str, object]:
        return {
            "name": "Test Character",
            "self_image": {
                "summary": "steady",
                "recent_window": ["first; second"],
                "meta": {
                    "last_updated": "2026-06-19T00:00:00+00:00",
                    "synthesis_count": 2,
                },
                "prompt_text": "must redact",
            },
        }

    async def empty_runtime_state() -> dict[str, object]:
        return {"reflection_summary": "background learning only"}

    async def mixed_growth_traits(*, limit: int) -> list[object]:
        assert limit == 12
        return [
            ["invalid"],
            {
                "trait_id": "trait-1",
                "trait_name": "observable growth",
                "prompt_text": "must redact",
            },
        ]

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        get_character_runtime_state=empty_runtime_state,
        list_growth_traits=mixed_growth_traits,
    )

    character = await repository.character_entity(limit=10)
    self_image = character["panels"]["self_image"]["items"][0]
    learning = character["panels"]["learning"]["items"][0]
    growth = character["panels"]["growth"]["items"][0]

    assert character["status"] == "available"
    assert self_image["summary"] == "steady"
    assert self_image["recent_window"] == ["first; second"]
    assert self_image["last_updated"] == "2026-06-19T00:00:00+00:00"
    assert self_image["synthesis_count"] == 2
    assert learning["source"] == "character_state.reflection_summary"
    assert growth["trait_id"] == "trait-1"
    rendered = repr(character).lower()
    assert "prompt_text" not in rendered
    assert "must redact" not in rendered


def test_stream_parsing_replay_and_status_events_cover_failure_branches() -> None:
    """Log streaming helpers should expose gaps and reject invalid filters."""

    from datetime import datetime, timezone

    from control_console.contracts import ProcessLogLine
    from control_console.stream import (
        LogStreamHub,
        SSEEventBuffer,
        encode_sse_event,
        log_keepalive_event,
        log_snapshot_event,
        log_status_event,
        parse_log_streams,
    )

    empty_buffer = SSEEventBuffer(max_events=1)
    buffer = SSEEventBuffer(max_events=2)
    buffer.append("control.first", {"line": "first"})
    second_id = buffer.append("control.second", {"line": "second"})
    replay = buffer.replay_after(second_id)
    empty_gap = empty_buffer.replay_after("1")
    non_numeric_gap = buffer.replay_after("not-a-number")

    hub = LogStreamHub(max_events=2)
    first_event = hub.publish(service_id="brain", stream="stdout", line="brain")
    hub.publish(service_id="adapter.debug", stream="stderr", line="adapter")
    after_first = hub.replay_after(
        cursor=first_event.cursor,
        service_id="all",
        streams={"stdout", "stderr"},
        tail=10,
    )
    no_match_replay = hub.replay_after(
        cursor="missing",
        service_id="brain",
        streams={"stderr"},
        tail=10,
    )
    all_tail = hub.replay_after(
        cursor=None,
        service_id="all",
        streams={"stdout", "stderr"},
        tail=2,
    )
    subscriber = hub.subscribe(service_id="brain", streams={"stderr"})
    hub.publish(service_id="adapter.debug", stream="stderr", line="filtered")
    status_event = log_status_event(
        service_id="brain",
        status="unavailable",
        message="logs unavailable",
    )
    keepalive = log_keepalive_event()
    encoded = encode_sse_event(status_event)
    snapshot = log_snapshot_event(
        ProcessLogLine(
            service_id="brain",
            stream="stdout",
            line="short line",
            created_at=datetime.now(timezone.utc),
            cursor="snapshot-1",
        ),
    )

    assert empty_gap[0].event_type == "control.gap"
    assert replay == []
    assert non_numeric_gap[0].event_type == "control.gap"
    assert after_first[0].data["service_id"] == "adapter.debug"
    assert no_match_replay == []
    assert [event.data["service_id"] for event in all_tail] == [
        "brain",
        "adapter.debug",
    ]
    assert subscriber.queue.empty()
    assert status_event.data["message"] == "logs unavailable"
    assert keepalive.event_type == "log.keepalive"
    assert snapshot.data["truncated"] is False
    assert parse_log_streams("") == {"stdout", "stderr", "supervisor"}
    assert parse_log_streams("stdout, stderr") == {"stdout", "stderr"}
    assert "event: log.status" in encoded
    with pytest.raises(ValueError, match="unknown log streams"):
        parse_log_streams("stdout,debug")


@pytest.mark.asyncio
async def test_app_service_helpers_cover_mixed_state_and_event_filters(tmp_path) -> None:
    """App helpers should serialize mixed service state and filtered logs."""

    from fastapi import HTTPException

    from control_console.app import (
        _brain_http_available,
        _event_matches_filters,
        _read_process_events,
        _service_actual_state,
        _service_config_snapshot_or_http_error,
        _service_config_summaries,
        _service_last_error_preview,
        _service_state_record,
        _service_state_for_debug,
        _service_state_payload,
        _service_version,
    )
    from control_console.contracts import OperationalEventQuery
    from control_console.log_store import ProcessLogStore
    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigOverrideStore,
        ServiceConfigRegistry,
    )
    from control_console.supervisor import ENDPOINT_CONFLICT_MESSAGE

    class ModelState:
        id = "brain"
        actual_state = "conflict"
        last_error_preview = ENDPOINT_CONFLICT_MESSAGE

        def model_dump(self, *, mode: str) -> dict[str, str]:
            assert mode == "json"
            return {
                "id": self.id,
                "actual_state": self.actual_state,
                "last_error_preview": self.last_error_preview,
            }

    class AsyncVersionSupervisor:
        async def service_version(self, service_id: str) -> int:
            assert service_id == "brain"
            return 9

    descriptor = ServiceConfigDescriptor(
        service_id="brain",
        title="Brain",
        description="Brain config.",
        fields=[
            ServiceConfigField(
                key="mode",
                label="Mode",
                description="Mode value.",
                value_type="enum",
                default_env="BRAIN_MODE",
                options=["safe"],
            ),
        ],
    )
    registry = ServiceConfigRegistry(descriptors=[descriptor])
    store = ProcessLogStore(tmp_path / "logs")
    store.append_line(service_id="brain", stream="stdout", line="startup")
    store.append_line(service_id="adapter.debug", stream="stderr", line="error")
    query = OperationalEventQuery(service_id="brain", limit=10)

    summaries = _service_config_summaries(
        states=[{"id": "missing"}, ModelState()],
        registry=registry,
        environment={"BRAIN_MODE": "invalid"},
        overrides=ServiceConfigOverrideStore(),
    )
    rows = await _read_process_events(
        query,
        log_store=store,
        services={"brain": object(), "adapter.debug": object()},
    )
    filtered_out = _event_matches_filters(
        {"service_id": "brain", "event_type": "service_started"},
        OperationalEventQuery(service_id="adapter.debug"),
    )
    version = await _service_version(
        supervisor=AsyncVersionSupervisor(),
        service_id="brain",
    )

    with pytest.raises(HTTPException) as missing_service_error:
        _service_config_snapshot_or_http_error(
            service_id="missing",
            services={"brain": object()},
            registry=registry,
            overrides=ServiceConfigOverrideStore(),
        )
    with pytest.raises(HTTPException) as missing_descriptor_error:
        _service_config_snapshot_or_http_error(
            service_id="adapter.debug",
            services={"adapter.debug": object()},
            registry=registry,
            overrides=ServiceConfigOverrideStore(),
        )
    with pytest.raises(HTTPException) as invalid_default_error:
        _service_config_snapshot_or_http_error(
            service_id="brain",
            services={"brain": object()},
            registry=registry,
            overrides=ServiceConfigOverrideStore(),
        )

    assert summaries["brain"] == {
        "configurable": True,
        "state": "unavailable",
        "apply_behavior": "restart",
        "field_count": 0,
    }
    assert missing_service_error.value.status_code == 404
    assert missing_descriptor_error.value.status_code == 404
    assert invalid_default_error.value.status_code == 422
    assert rows[0]["service_id"] == "brain"
    assert rows[0]["message"] == "startup"
    assert filtered_out is False
    assert version == 9
    assert _service_state_payload(None) == {"actual_state": "unavailable"}
    assert _service_state_payload({"actual_state": "running"}) == {
        "actual_state": "running",
    }
    assert _service_state_payload(ModelState())["actual_state"] == "conflict"
    assert _service_state_payload("broken") == {"actual_state": "broken"}
    assert _service_actual_state(None) == "unavailable"
    assert _service_actual_state({"actual_state": "running"}) == "running"
    assert _service_actual_state(ModelState()) == "conflict"
    assert _service_last_error_preview(None) == ""
    assert _service_last_error_preview({"last_error_preview": "bad"}) == "bad"
    assert _service_last_error_preview(ModelState()) == ENDPOINT_CONFLICT_MESSAGE
    assert _brain_http_available("running") is True
    assert _brain_http_available(
        "conflict",
        last_error_preview=ENDPOINT_CONFLICT_MESSAGE,
    ) is True
    assert _brain_http_available("stopped") is False
    assert _service_state_record([{"id": "adapter.debug"}], service_id="brain") is None
    assert _service_state_for_debug(object(), {}) is None
