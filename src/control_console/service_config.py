"""Descriptor-driven runtime service configuration for the control console."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


NAPCAT_ACTIVE_GROUP_PATTERN = r"^[0-9]{1,32}$"
DEFAULT_MAX_ITEMS = 50
DEFAULT_MAX_ITEM_LENGTH = 120
MAX_DESCRIPTOR_FIELDS = 64

ConfigValue = str | int | bool | list[str]
CommandRenderer = Callable[[list[str], dict[str, object]], list[str]]
EnvironmentRenderer = Callable[[dict[str, object]], dict[str, str]]


class ServiceConfigValidationError(ValueError):
    """Raised when descriptor values cannot be validated or projected."""


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown descriptor fields."""

    model_config = ConfigDict(extra="forbid")


class ServiceConfigField(StrictConfigModel):
    """One operator-editable field in a service configuration descriptor."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)
    value_type: Literal["string_list", "string", "boolean", "integer", "enum"]
    default_env: str | None = Field(default=None, pattern=r"^[A-Z0-9_]{1,80}$")
    default_fallback_env: list[str] = Field(default_factory=list, max_length=8)
    default_literal: str = Field(default="", max_length=240)
    sensitive: bool = False
    restart_required: bool = True
    max_items: int = Field(default=DEFAULT_MAX_ITEMS, ge=1, le=500)
    max_item_length: int = Field(default=DEFAULT_MAX_ITEM_LENGTH, ge=1, le=1000)
    pattern: str | None = Field(default=None, max_length=240)
    options: list[str] = Field(default_factory=list, max_length=64)
    min_value: int | None = None
    max_value: int | None = None

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, pattern: str | None) -> str | None:
        """Compile configured regex patterns at descriptor load time."""

        if pattern is None:
            return None
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid field pattern: {exc}") from exc
        return pattern

    @field_validator("default_fallback_env")
    @classmethod
    def _validate_default_fallback_env(cls, values: list[str]) -> list[str]:
        """Reject non-environment fallback names in descriptors."""

        for value in values:
            if re.fullmatch(r"^[A-Z0-9_]{1,80}$", value) is None:
                raise ValueError("fallback environment keys must be uppercase names")
        return values


class ServiceConfigDescriptor(StrictConfigModel):
    """Configuration descriptor registered for one service id."""

    service_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    title: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)
    fields: list[ServiceConfigField] = Field(
        min_length=1,
        max_length=MAX_DESCRIPTOR_FIELDS,
    )

    @field_validator("fields")
    @classmethod
    def _validate_unique_fields(
        cls,
        fields: list[ServiceConfigField],
    ) -> list[ServiceConfigField]:
        """Reject descriptors that define the same field key more than once."""

        seen_keys: set[str] = set()
        for field in fields:
            if field.key in seen_keys:
                raise ValueError(f"duplicate config field: {field.key}")
            seen_keys.add(field.key)
        return fields


class ServiceConfigFieldSnapshot(StrictConfigModel):
    """Browser-safe snapshot for one descriptor field."""

    key: str
    label: str
    description: str
    value_type: str
    default_source: str | None = None
    default_value: Any = None
    override_value: Any = None
    effective_value: Any = None
    restart_required: bool
    sensitive: bool
    validation: dict[str, Any] = Field(default_factory=dict)


class ServiceConfigSnapshot(StrictConfigModel):
    """Browser-safe configuration snapshot for one service."""

    service_id: str
    title: str
    description: str
    apply_behavior: Literal["restart"] = "restart"
    state: Literal["default", "override_active", "apply_failed", "unavailable"]
    fields: list[ServiceConfigFieldSnapshot]


class ServiceConfigOverrideStore:
    """Process-local service configuration overrides."""

    def __init__(self) -> None:
        """Create an empty in-memory override store."""

        self._overrides: dict[str, dict[str, ConfigValue]] = {}
        self._apply_failed_services: set[str] = set()

    def override_for_service(self, service_id: str) -> dict[str, ConfigValue]:
        """Return a copy of the current override for one service."""

        override = dict(self._overrides.get(service_id, {}))
        return override

    def set_override(
        self,
        service_id: str,
        values: Mapping[str, object],
        registry: "ServiceConfigRegistry",
        environment: Mapping[str, str],
    ) -> None:
        """Validate and store a process-local override for one service."""

        validated_values = registry.validate_values(
            service_id=service_id,
            values=values,
            environment=environment,
        )
        self._overrides[service_id] = validated_values

    def clear_override(self, service_id: str) -> None:
        """Remove the process-local override for one service."""

        self._overrides.pop(service_id, None)

    def mark_apply_failed(self, service_id: str) -> None:
        """Mark the latest restart-based apply attempt as failed."""

        self._apply_failed_services.add(service_id)

    def clear_apply_failed(self, service_id: str) -> None:
        """Clear any previous restart failure marker for one service."""

        self._apply_failed_services.discard(service_id)

    def apply_failed(self, service_id: str) -> bool:
        """Return whether the latest apply attempt failed for one service."""

        failed = service_id in self._apply_failed_services
        return failed


class ServiceConfigRegistry:
    """Registry of service descriptors and command renderers."""

    def __init__(
        self,
        *,
        descriptors: list[ServiceConfigDescriptor],
        command_renderers: dict[str, CommandRenderer] | None = None,
        environment_renderers: dict[str, EnvironmentRenderer] | None = None,
    ) -> None:
        """Create a registry from descriptors and optional command renderers."""

        self._descriptors: dict[str, ServiceConfigDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.service_id in self._descriptors:
                raise ServiceConfigValidationError(
                    f"duplicate service descriptor: {descriptor.service_id}"
                )
            self._descriptors[descriptor.service_id] = descriptor
        self._command_renderers = dict(command_renderers or {})
        self._environment_renderers = dict(environment_renderers or {})

    def has_descriptor(self, service_id: str) -> bool:
        """Return whether a service has an operator config descriptor."""

        has_descriptor = service_id in self._descriptors
        return has_descriptor

    def configurable_service_ids(self) -> list[str]:
        """Return service ids with registered descriptors."""

        service_ids = list(self._descriptors)
        return service_ids

    def snapshot_for_service(
        self,
        service_id: str,
        environment: Mapping[str, str],
        overrides: ServiceConfigOverrideStore,
    ) -> ServiceConfigSnapshot:
        """Project default, override, and effective values for one service."""

        descriptor = self._descriptor_for_service(service_id)
        override_values = overrides.override_for_service(service_id)
        fields: list[ServiceConfigFieldSnapshot] = []
        for field in descriptor.fields:
            default_value = self._default_value(field=field, environment=environment)
            override_value = override_values.get(field.key)
            if field.key in override_values:
                effective_value = override_value
            else:
                effective_value = default_value
            snapshot = ServiceConfigFieldSnapshot(
                key=field.key,
                label=field.label,
                description=field.description,
                value_type=field.value_type,
                default_source=field.default_env,
                default_value=_visible_value(field, default_value),
                override_value=_visible_value(field, override_value),
                effective_value=_visible_value(field, effective_value),
                restart_required=field.restart_required,
                sensitive=field.sensitive,
                validation=_validation_metadata(field),
            )
            fields.append(snapshot)

        state: Literal["default", "override_active", "apply_failed"] = "default"
        if overrides.apply_failed(service_id):
            state = "apply_failed"
        elif override_values:
            state = "override_active"
        config_snapshot = ServiceConfigSnapshot(
            service_id=descriptor.service_id,
            title=descriptor.title,
            description=descriptor.description,
            state=state,
            fields=fields,
        )
        return config_snapshot

    def validate_values(
        self,
        *,
        service_id: str,
        values: Mapping[str, object],
        environment: Mapping[str, str],
    ) -> dict[str, ConfigValue]:
        """Validate operator-submitted values against one descriptor."""

        descriptor = self._descriptor_for_service(service_id)
        field_by_key = {field.key: field for field in descriptor.fields}
        validated_values: dict[str, ConfigValue] = {}
        for key, value in values.items():
            field = field_by_key.get(key)
            if field is None:
                raise ServiceConfigValidationError(f"unknown config field: {key}")
            validated_value = _validate_value(field=field, value=value)
            validated_values[key] = validated_value

        _ = environment
        return validated_values

    def render_start_command(
        self,
        service_id: str,
        base_command: list[str],
        environment: Mapping[str, str],
        overrides: ServiceConfigOverrideStore,
    ) -> list[str]:
        """Render the argv used to start a service with effective overrides."""

        renderer = self._command_renderers.get(service_id)
        if renderer is None:
            rendered_command = list(base_command)
            return rendered_command

        snapshot = self.snapshot_for_service(
            service_id=service_id,
            environment=environment,
            overrides=overrides,
        )
        effective_values = {
            field.key: field.effective_value
            for field in snapshot.fields
        }
        rendered_command = renderer(list(base_command), effective_values)
        _validate_rendered_command(rendered_command)
        return rendered_command

    def render_environment_overlay(
        self,
        *,
        service_id: str,
        environment: Mapping[str, str],
        overrides: ServiceConfigOverrideStore,
    ) -> dict[str, str]:
        """Render descriptor-approved environment values for service start."""

        renderer = self._environment_renderers.get(service_id)
        if renderer is None:
            return {}

        snapshot = self.snapshot_for_service(
            service_id=service_id,
            environment=environment,
            overrides=overrides,
        )
        override_values = {
            field.key: field.override_value
            for field in snapshot.fields
            if field.override_value is not None
        }
        rendered_environment = renderer(override_values)
        _validate_environment_overlay(rendered_environment)
        return rendered_environment

    def _descriptor_for_service(self, service_id: str) -> ServiceConfigDescriptor:
        """Return one descriptor or raise a validation error."""

        descriptor = self._descriptors.get(service_id)
        if descriptor is None:
            raise ServiceConfigValidationError(
                f"service has no config descriptor: {service_id}"
            )
        return descriptor

    def _default_value(
        self,
        *,
        field: ServiceConfigField,
        environment: Mapping[str, str],
    ) -> ConfigValue:
        """Return the validated default value for one descriptor field."""

        raw_value = ""
        if field.default_env is not None:
            raw_value = environment.get(field.default_env, "")
        if not raw_value.strip():
            for fallback_env in field.default_fallback_env:
                raw_value = environment.get(fallback_env, "")
                if raw_value.strip():
                    break
        if not raw_value.strip():
            raw_value = field.default_literal
        default_value = _default_from_raw(field=field, raw_value=raw_value)
        return default_value


def build_default_service_config_registry() -> ServiceConfigRegistry:
    """Build production service config descriptors and command renderers."""

    brain_descriptor = ServiceConfigDescriptor(
        service_id="brain",
        title="Brain model routes",
        description="Restart-applied model routing for Brain chat LLM stages.",
        fields=_brain_model_route_fields(),
    )
    napcat_descriptor = ServiceConfigDescriptor(
        service_id="adapter.napcat",
        title="NapCat QQ adapter",
        description="Runtime configuration applied by restarting the service.",
        fields=[
            ServiceConfigField(
                key="active_groups",
                label="Active QQ groups",
                description="QQ groups where the adapter may visibly participate.",
                value_type="string_list",
                default_env="NAPCAT_ACTIVE_GROUPS",
                pattern=NAPCAT_ACTIVE_GROUP_PATTERN,
                max_items=DEFAULT_MAX_ITEMS,
                restart_required=True,
            ),
        ],
    )
    registry = ServiceConfigRegistry(
        descriptors=[brain_descriptor, napcat_descriptor],
        command_renderers={"adapter.napcat": _render_napcat_command},
        environment_renderers={"brain": _render_brain_environment},
    )
    return registry


def _default_from_raw(
    *,
    field: ServiceConfigField,
    raw_value: str,
) -> ConfigValue:
    """Parse and validate an environment-backed default value."""

    if field.value_type == "string_list":
        raw_items = re.split(r"[\s,]+", raw_value.strip()) if raw_value.strip() else []
        value = [item for item in raw_items if item]
    elif field.value_type == "boolean":
        value = _boolean_from_raw(field=field, raw_value=raw_value)
    elif field.value_type == "integer":
        value = _integer_from_raw(field=field, raw_value=raw_value)
    else:
        value = raw_value

    default_value = _validate_value(field=field, value=value)
    return default_value


def _boolean_from_raw(*, field: ServiceConfigField, raw_value: str) -> bool:
    """Parse a boolean default from an environment string."""

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"", "0", "false", "no", "off"}:
        return False
    raise ServiceConfigValidationError(
        f"{field.key}: default environment value must be boolean"
    )


def _integer_from_raw(*, field: ServiceConfigField, raw_value: str) -> int:
    """Parse an integer default from an environment string."""

    normalized_value = raw_value.strip()
    if not normalized_value:
        return 0
    try:
        parsed_value = int(normalized_value)
    except ValueError as exc:
        raise ServiceConfigValidationError(
            f"{field.key}: default environment value must be integer"
        ) from exc
    return parsed_value


def _validate_value(*, field: ServiceConfigField, value: object) -> ConfigValue:
    """Validate one field value from defaults or operator input."""

    if field.value_type == "string_list":
        validated_value = _validate_string_list(field=field, value=value)
        return validated_value
    if field.value_type == "boolean":
        validated_value = _validate_boolean(field=field, value=value)
        return validated_value
    if field.value_type == "integer":
        validated_value = _validate_integer(field=field, value=value)
        return validated_value

    validated_value = _validate_string(field=field, value=value)
    return validated_value


def _validate_string_list(
    *,
    field: ServiceConfigField,
    value: object,
) -> list[str]:
    """Validate a list of bounded string argv parts."""

    if not isinstance(value, list):
        raise ServiceConfigValidationError(f"{field.key}: value must be a list")
    if len(value) > field.max_items:
        raise ServiceConfigValidationError(
            f"{field.key}: value contains too many items"
        )

    validated_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ServiceConfigValidationError(
                f"{field.key}: list items must be strings"
            )
        normalized_item = item.strip()
        _validate_string_constraints(field=field, value=normalized_item)
        validated_items.append(normalized_item)
    return validated_items


def _validate_boolean(*, field: ServiceConfigField, value: object) -> bool:
    """Validate a boolean config value."""

    if not isinstance(value, bool):
        raise ServiceConfigValidationError(f"{field.key}: value must be boolean")
    return value


def _validate_integer(*, field: ServiceConfigField, value: object) -> int:
    """Validate an integer config value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceConfigValidationError(f"{field.key}: value must be integer")
    if field.min_value is not None and value < field.min_value:
        raise ServiceConfigValidationError(
            f"{field.key}: value must be at least {field.min_value}"
        )
    if field.max_value is not None and value > field.max_value:
        raise ServiceConfigValidationError(
            f"{field.key}: value must be at most {field.max_value}"
        )
    return value


def _validate_string(*, field: ServiceConfigField, value: object) -> str:
    """Validate a string or enum config value."""

    if not isinstance(value, str):
        raise ServiceConfigValidationError(f"{field.key}: value must be string")
    normalized_value = value.strip()
    _validate_string_constraints(field=field, value=normalized_value)
    if field.value_type == "enum" and field.options:
        if normalized_value not in field.options:
            raise ServiceConfigValidationError(
                f"{field.key}: value must be one of the descriptor options"
            )
    return normalized_value


def _validate_string_constraints(
    *,
    field: ServiceConfigField,
    value: str,
) -> None:
    """Apply shared string length and pattern constraints."""

    if not value:
        raise ServiceConfigValidationError(f"{field.key}: value must not be empty")
    if len(value) > field.max_item_length:
        raise ServiceConfigValidationError(f"{field.key}: value is too long")
    if field.pattern is not None and re.fullmatch(field.pattern, value) is None:
        raise ServiceConfigValidationError(
            f"{field.key}: value does not match required pattern"
        )


def _validation_metadata(field: ServiceConfigField) -> dict[str, Any]:
    """Return browser-safe validation metadata for one field."""

    validation: dict[str, Any] = {}
    if field.pattern is not None:
        validation["pattern"] = field.pattern
    if field.value_type == "string_list":
        validation["max_items"] = field.max_items
        validation["max_item_length"] = field.max_item_length
    if field.value_type == "enum":
        validation["options"] = list(field.options)
    if field.value_type == "integer":
        if field.min_value is not None:
            validation["min_value"] = field.min_value
        if field.max_value is not None:
            validation["max_value"] = field.max_value
    return validation


def _visible_value(field: ServiceConfigField, value: object) -> object:
    """Hide sensitive field values from snapshots."""

    if field.sensitive and value is not None:
        return None
    return value


def _render_napcat_command(
    base_command: list[str],
    effective_values: dict[str, object],
) -> list[str]:
    """Render NapCat active groups into the existing adapter argv shape."""

    active_groups = effective_values["active_groups"]
    if not isinstance(active_groups, list):
        raise ServiceConfigValidationError("active_groups: value must be a list")
    if not active_groups:
        rendered_command = list(base_command)
        return rendered_command

    rendered_command = [*base_command, "--channels", *active_groups]
    return rendered_command


def _brain_model_route_fields() -> list[ServiceConfigField]:
    """Return descriptor fields for every Brain model route."""

    from control_console.brain_model_routes import (
        MODEL_ID_PATTERN,
        route_default_fallback_env,
        route_default_literal,
        route_descriptors,
        route_env_name,
        route_field_key,
    )

    fields: list[ServiceConfigField] = []
    for route in route_descriptors():
        fields.extend([
            ServiceConfigField(
                key=route_field_key(route, "model"),
                label=f"{route.label} model",
                description=f"Model ID for the {route.label} Brain route.",
                value_type="string",
                default_env=route_env_name(route, "model"),
                default_fallback_env=route_default_fallback_env(route, "model"),
                default_literal=route_default_literal("model"),
                pattern=MODEL_ID_PATTERN,
                max_item_length=200,
                restart_required=True,
            ),
            ServiceConfigField(
                key=route_field_key(route, "max_completion_tokens"),
                label=f"{route.label} max completion tokens",
                description=(
                    f"Maximum completion token budget for the {route.label} route."
                ),
                value_type="integer",
                default_env=route_env_name(route, "max_completion_tokens"),
                default_fallback_env=route_default_fallback_env(
                    route,
                    "max_completion_tokens",
                ),
                default_literal=route_default_literal("max_completion_tokens"),
                min_value=1,
                max_value=65536,
                restart_required=True,
            ),
            ServiceConfigField(
                key=route_field_key(route, "thinking_enabled"),
                label=f"{route.label} thinking mode",
                description=f"Whether the {route.label} route enables thinking.",
                value_type="boolean",
                default_env=route_env_name(route, "thinking_enabled"),
                default_fallback_env=route_default_fallback_env(
                    route,
                    "thinking_enabled",
                ),
                default_literal=route_default_literal("thinking_enabled"),
                restart_required=True,
            ),
        ])
    return fields


def _render_brain_environment(
    effective_values: dict[str, object],
) -> dict[str, str]:
    """Render Brain route descriptor values into child-process env vars."""

    from control_console.brain_model_routes import (
        route_descriptors,
        route_env_name,
        route_field_key,
    )

    environment: dict[str, str] = {}
    for route in route_descriptors():
        for field_name in route.editable_fields:
            key = route_field_key(route, field_name)
            if key not in effective_values:
                continue
            value = effective_values[key]
            env_name = route_env_name(route, field_name)
            environment[env_name] = _environment_value(key=key, value=value)
    return environment


def _environment_value(*, key: str, value: object) -> str:
    """Serialize a validated descriptor value for process environment use."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    raise ServiceConfigValidationError(f"{key}: value cannot be rendered to env")


def _validate_rendered_command(command: list[str]) -> None:
    """Reject command renderers that return non-argv values."""

    if not isinstance(command, list) or not command:
        raise ServiceConfigValidationError("rendered command must be a non-empty list")
    for part in command:
        if not isinstance(part, str) or not part.strip():
            raise ServiceConfigValidationError(
                "rendered command entries must be non-empty strings"
            )


def _validate_environment_overlay(environment: dict[str, str]) -> None:
    """Reject malformed environment overlays from descriptor renderers."""

    for key, value in environment.items():
        if re.fullmatch(r"^[A-Z0-9_]{1,80}$", key) is None:
            raise ServiceConfigValidationError(
                "rendered environment keys must be uppercase names"
            )
        if not isinstance(value, str):
            raise ServiceConfigValidationError(
                "rendered environment values must be strings"
            )
        if len(value) > 2000:
            raise ServiceConfigValidationError(
                "rendered environment values are too large"
            )
