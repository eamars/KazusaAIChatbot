"""Service-configuration contract tests for the control console."""

from __future__ import annotations

import pytest


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
