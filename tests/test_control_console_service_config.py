"""Service-configuration contract tests for the control console."""

from __future__ import annotations

import pytest


def _brain_service_environment() -> dict[str, str]:
    """Provide valid route defaults while testing non-route Brain config."""

    from control_console.brain_model_routes import (
        route_descriptors,
        route_env_name,
    )

    environment: dict[str, str] = {}
    for route in route_descriptors():
        environment[route_env_name(route, "model")] = "test-model"
        environment[route_env_name(route, "max_completion_tokens")] = "8192"
        environment[route_env_name(route, "thinking_enabled")] = "false"
    return environment


def test_obsolete_background_route_descriptor_is_absent() -> None:
    """The console catalog keeps worker controls without the retired route."""

    from control_console.brain_model_routes import route_descriptors

    route_keys = {route.route_key for route in route_descriptors()}
    assert "background_work" not in route_keys


def test_snapshot_uses_environment_default_and_ephemeral_override() -> None:
    """A descriptor snapshot separates default, override, and effective values."""

    from control_console.service_config import (
        ServiceConfigOverrideStore,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    overrides = ServiceConfigOverrideStore()
    environment = {"NAPCAT_ACTIVE_GROUPS": "54369546, 905393941"}

    default_snapshot = registry.snapshot_for_service(
        service_id="adapter.napcat",
        environment=environment,
        overrides=overrides,
    )

    assert default_snapshot.service_id == "adapter.napcat"
    assert default_snapshot.state == "default"
    assert default_snapshot.apply_behavior == "restart"
    assert default_snapshot.fields[0].key == "active_groups"
    assert default_snapshot.fields[0].default_source == "NAPCAT_ACTIVE_GROUPS"
    assert default_snapshot.fields[0].default_value == ["54369546", "905393941"]
    assert default_snapshot.fields[0].override_value is None
    assert default_snapshot.fields[0].effective_value == [
        "54369546",
        "905393941",
    ]

    overrides.set_override(
        service_id="adapter.napcat",
        values={"active_groups": ["112233"]},
        registry=registry,
        environment=environment,
    )
    override_snapshot = registry.snapshot_for_service(
        service_id="adapter.napcat",
        environment=environment,
        overrides=overrides,
    )

    assert override_snapshot.state == "override_active"
    assert override_snapshot.fields[0].default_value == ["54369546", "905393941"]
    assert override_snapshot.fields[0].override_value == ["112233"]
    assert override_snapshot.fields[0].effective_value == ["112233"]

    fresh_overrides = ServiceConfigOverrideStore()
    fresh_snapshot = registry.snapshot_for_service(
        service_id="adapter.napcat",
        environment=environment,
        overrides=fresh_overrides,
    )

    assert fresh_snapshot.state == "default"
    assert fresh_snapshot.fields[0].override_value is None
    assert fresh_snapshot.fields[0].effective_value == ["54369546", "905393941"]


def test_reset_clears_override_back_to_environment_default() -> None:
    """Reset removes only the process-local override for the selected service."""

    from control_console.service_config import (
        ServiceConfigOverrideStore,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    overrides = ServiceConfigOverrideStore()
    environment = {"NAPCAT_ACTIVE_GROUPS": "54369546"}

    overrides.set_override(
        service_id="adapter.napcat",
        values={"active_groups": ["905393941"]},
        registry=registry,
        environment=environment,
    )
    overrides.clear_override(service_id="adapter.napcat")

    snapshot = registry.snapshot_for_service(
        service_id="adapter.napcat",
        environment=environment,
        overrides=overrides,
    )

    assert snapshot.state == "default"
    assert snapshot.fields[0].override_value is None
    assert snapshot.fields[0].effective_value == ["54369546"]


def test_validation_rejects_invalid_napcat_active_groups() -> None:
    """Group ids must be bounded numeric argv parts before command rendering."""

    from control_console.service_config import (
        ServiceConfigOverrideStore,
        ServiceConfigValidationError,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    overrides = ServiceConfigOverrideStore()
    environment = {"NAPCAT_ACTIVE_GROUPS": ""}

    invalid_values = [
        {"active_groups": ["abc"]},
        {"active_groups": ["123", "456 789"]},
        {"active_groups": ["1" * 33]},
        {"active_groups": ["1"] * 51},
    ]

    for values in invalid_values:
        with pytest.raises(ServiceConfigValidationError, match="active_groups"):
            overrides.set_override(
                service_id="adapter.napcat",
                values=values,
                registry=registry,
                environment=environment,
            )


def test_fake_non_napcat_descriptor_uses_same_snapshot_and_validation_path() -> None:
    """A test-only service descriptor proves the API is not NapCat-shaped."""

    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigOverrideStore,
        ServiceConfigRegistry,
        ServiceConfigValidationError,
    )

    descriptor = ServiceConfigDescriptor(
        service_id="adapter.fake",
        title="Fake adapter",
        description="Test-only adapter descriptor.",
        fields=[
            ServiceConfigField(
                key="enabled",
                label="Enabled",
                description="Whether the fake adapter participates.",
                value_type="boolean",
                default_env="FAKE_ADAPTER_ENABLED",
                restart_required=True,
            ),
        ],
    )
    registry = ServiceConfigRegistry(descriptors=[descriptor])
    overrides = ServiceConfigOverrideStore()
    environment = {"FAKE_ADAPTER_ENABLED": "true"}

    default_snapshot = registry.snapshot_for_service(
        service_id="adapter.fake",
        environment=environment,
        overrides=overrides,
    )
    assert default_snapshot.service_id == "adapter.fake"
    assert default_snapshot.state == "default"
    assert default_snapshot.fields[0].key == "enabled"
    assert default_snapshot.fields[0].default_value is True
    assert default_snapshot.fields[0].effective_value is True

    overrides.set_override(
        service_id="adapter.fake",
        values={"enabled": False},
        registry=registry,
        environment=environment,
    )
    override_snapshot = registry.snapshot_for_service(
        service_id="adapter.fake",
        environment=environment,
        overrides=overrides,
    )
    assert override_snapshot.state == "override_active"
    assert override_snapshot.fields[0].override_value is False
    assert override_snapshot.fields[0].effective_value is False

    with pytest.raises(ServiceConfigValidationError, match="enabled"):
        overrides.set_override(
            service_id="adapter.fake",
            values={"enabled": "false"},
            registry=registry,
            environment=environment,
        )


def test_command_renderers_are_generic_and_append_napcat_channels() -> None:
    """Command overlays render from descriptors without exposing shell strings."""

    from control_console.service_config import (
        ServiceConfigDescriptor,
        ServiceConfigField,
        ServiceConfigOverrideStore,
        ServiceConfigRegistry,
        build_default_service_config_registry,
    )

    napcat_registry = build_default_service_config_registry()
    napcat_overrides = ServiceConfigOverrideStore()
    napcat_base_command = [
        "python",
        "-m",
        "adapters.napcat_qq_adapter",
    ]

    default_command = napcat_registry.render_start_command(
        service_id="adapter.napcat",
        base_command=napcat_base_command,
        environment={"NAPCAT_ACTIVE_GROUPS": ""},
        overrides=napcat_overrides,
    )
    assert default_command == napcat_base_command

    napcat_overrides.set_override(
        service_id="adapter.napcat",
        values={"active_groups": ["54369546", "905393941"]},
        registry=napcat_registry,
        environment={"NAPCAT_ACTIVE_GROUPS": ""},
    )
    override_command = napcat_registry.render_start_command(
        service_id="adapter.napcat",
        base_command=napcat_base_command,
        environment={"NAPCAT_ACTIVE_GROUPS": ""},
        overrides=napcat_overrides,
    )
    assert override_command == [
        "python",
        "-m",
        "adapters.napcat_qq_adapter",
        "--channels",
        "54369546",
        "905393941",
    ]

    fake_descriptor = ServiceConfigDescriptor(
        service_id="adapter.fake",
        title="Fake adapter",
        description="Test-only adapter descriptor.",
        fields=[
            ServiceConfigField(
                key="enabled",
                label="Enabled",
                description="Whether the fake adapter participates.",
                value_type="boolean",
                default_env="FAKE_ADAPTER_ENABLED",
            ),
        ],
    )

    def fake_renderer(
        base_command: list[str],
        effective_values: dict[str, object],
    ) -> list[str]:
        enabled = effective_values["enabled"]
        rendered_command = [*base_command, "--enabled", str(enabled).lower()]
        return rendered_command

    fake_registry = ServiceConfigRegistry(
        descriptors=[fake_descriptor],
        command_renderers={"adapter.fake": fake_renderer},
    )
    fake_command = fake_registry.render_start_command(
        service_id="adapter.fake",
        base_command=["python", "-m", "adapter.fake"],
        environment={"FAKE_ADAPTER_ENABLED": "true"},
        overrides=ServiceConfigOverrideStore(),
    )

    assert fake_command == ["python", "-m", "adapter.fake", "--enabled", "true"]


def test_brain_config_exposes_identity_growth_pace_and_renders_overrides() -> None:
    """All five bounded pace controls should restart-apply to Brain."""

    from control_console.service_config import (
        ServiceConfigOverrideStore,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    overrides = ServiceConfigOverrideStore()
    environment = _brain_service_environment()

    snapshot = registry.snapshot_for_service(
        service_id="brain",
        environment=environment,
        overrides=overrides,
    )
    fields = {field.key: field for field in snapshot.fields}
    expected = {
        "character_identity_growth_enabled": (
            "CHARACTER_IDENTITY_GROWTH_ENABLED",
            True,
            {},
        ),
        "character_identity_growth_inferred_min_episodes": (
            "CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_EPISODES",
            3,
            {"min_value": 2, "max_value": 8},
        ),
        "character_identity_growth_inferred_min_local_dates": (
            "CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_LOCAL_DATES",
            2,
            {"min_value": 1, "max_value": 7},
        ),
        "character_identity_growth_max_inferred_promotions_per_local_day": (
            "CHARACTER_IDENTITY_GROWTH_MAX_INFERRED_PROMOTIONS_PER_LOCAL_DAY",
            1,
            {"min_value": 0, "max_value": 3},
        ),
        "character_identity_growth_prompt_char_budget": (
            "CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET",
            18_000,
            {"min_value": 8_000, "max_value": 30_000},
        ),
    }

    for field_key, (
        environment_name,
        default_value,
        validation,
    ) in expected.items():
        field = fields[field_key]
        assert field.default_source == environment_name
        assert field.default_value == default_value
        assert field.effective_value == default_value
        assert field.validation == validation
        assert field.restart_required is True

    overrides.set_override(
        service_id="brain",
        values={
            "character_identity_growth_enabled": False,
            "character_identity_growth_inferred_min_episodes": 6,
            "character_identity_growth_inferred_min_local_dates": 4,
            "character_identity_growth_max_inferred_promotions_per_local_day": 2,
            "character_identity_growth_prompt_char_budget": 24_000,
        },
        registry=registry,
        environment=environment,
    )
    overlay = registry.render_environment_overlay(
        service_id="brain",
        environment=environment,
        overrides=overrides,
    )

    assert overlay == {
        "CHARACTER_IDENTITY_GROWTH_ENABLED": "false",
        "CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_EPISODES": "6",
        "CHARACTER_IDENTITY_GROWTH_INFERRED_MIN_LOCAL_DATES": "4",
        "CHARACTER_IDENTITY_GROWTH_MAX_INFERRED_PROMOTIONS_PER_LOCAL_DAY": "2",
        "CHARACTER_IDENTITY_GROWTH_PROMPT_CHAR_BUDGET": "24000",
    }


def test_brain_identity_growth_pace_rejects_bounds_and_cross_field_values() -> None:
    """Pace settings should fail before an invalid Brain restart."""

    from control_console.service_config import (
        ServiceConfigOverrideStore,
        ServiceConfigValidationError,
        build_default_service_config_registry,
    )

    registry = build_default_service_config_registry()
    invalid_values = (
        {"character_identity_growth_inferred_min_episodes": 1},
        {"character_identity_growth_inferred_min_episodes": 9},
        {"character_identity_growth_inferred_min_local_dates": 0},
        {"character_identity_growth_inferred_min_local_dates": 8},
        {
            "character_identity_growth_max_inferred_promotions_per_local_day": -1,
        },
        {
            "character_identity_growth_max_inferred_promotions_per_local_day": 4,
        },
        {"character_identity_growth_prompt_char_budget": 7_999},
        {"character_identity_growth_prompt_char_budget": 30_001},
        {
            "character_identity_growth_inferred_min_episodes": 2,
            "character_identity_growth_inferred_min_local_dates": 3,
        },
    )

    for values in invalid_values:
        with pytest.raises(ServiceConfigValidationError):
            ServiceConfigOverrideStore().set_override(
                service_id="brain",
                values=values,
                registry=registry,
                environment=_brain_service_environment(),
            )
