"""Validated registry of console-managed local services."""

from __future__ import annotations

from pathlib import Path
import json
import sys

from pydantic import ValidationError

from control_console.contracts import ServiceSpec


class RegistryValidationError(ValueError):
    """Raised when a service registry cannot be safely loaded."""


def default_service_registry() -> dict[str, ServiceSpec]:
    """Return built-in services for the local Kazusa application."""

    python_executable = sys.executable
    specs = [
        ServiceSpec(
            id="brain",
            display_name="Brain service",
            kind="backend",
            command=[python_executable, "-m", "kazusa_ai_chatbot.main"],
            cwd=".",
            health_url="http://127.0.0.1:8000/health",
        ),
        ServiceSpec(
            id="adapter.discord",
            display_name="Discord adapter",
            kind="adapter",
            command=[python_executable, "-m", "adapters.discord_adapter"],
            cwd=".",
            dependencies=["brain"],
        ),
        ServiceSpec(
            id="adapter.napcat",
            display_name="NapCat QQ adapter",
            kind="adapter",
            command=[python_executable, "-m", "adapters.napcat_qq_adapter"],
            cwd=".",
            dependencies=["brain"],
        ),
        ServiceSpec(
            id="adapter.debug",
            display_name="Debug adapter",
            kind="adapter",
            command=[
                python_executable,
                "-m",
                "adapters.debug_adapter",
                "--brain-url",
                "http://127.0.0.1:8000",
                "--port",
                "8080",
            ],
            cwd=".",
            dependencies=["brain"],
            health_url="http://127.0.0.1:8080/api/health",
        ),
    ]
    registry = {spec.id: spec for spec in specs}
    return registry


def load_service_registry(
    *,
    override_path: Path | None,
    repo_root: Path,
) -> dict[str, ServiceSpec]:
    """Load defaults or a JSON override after strict validation."""

    if override_path is None:
        registry = default_service_registry()
        return registry

    try:
        raw_text = override_path.read_text(encoding="utf-8")
        raw_document = json.loads(raw_text)
    except OSError as exc:
        raise RegistryValidationError(f"cannot read service registry: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(f"service registry is not JSON: {exc}") from exc

    raw_services = raw_document.get("services")
    if not isinstance(raw_services, list):
        raise RegistryValidationError("service registry must contain a services list")

    services: dict[str, ServiceSpec] = {}
    for raw_service in raw_services:
        try:
            service = ServiceSpec.model_validate(raw_service)
        except ValidationError as exc:
            raise RegistryValidationError(f"invalid service spec: {exc}") from exc
        if service.id in services:
            raise RegistryValidationError(f"duplicate service id: {service.id}")
        _validate_cwd(service=service, repo_root=repo_root)
        services[service.id] = service

    _validate_dependencies(services)
    return services


def _validate_cwd(*, service: ServiceSpec, repo_root: Path) -> None:
    """Reject service working directories outside the repository."""

    if service.cwd is None:
        return

    root = repo_root.resolve()
    candidate = (root / service.cwd).resolve()
    if root != candidate and root not in candidate.parents:
        raise RegistryValidationError(f"cwd escapes repository: {service.id}")


def _validate_dependencies(services: dict[str, ServiceSpec]) -> None:
    """Validate dependency references and cycles."""

    for service in services.values():
        for dependency in service.dependencies:
            if dependency not in services:
                raise RegistryValidationError(
                    f"unknown dependency {dependency} for {service.id}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(service_id: str) -> None:
        if service_id in visited:
            return
        if service_id in visiting:
            raise RegistryValidationError("service dependency cycle detected")
        visiting.add(service_id)
        service = services[service_id]
        for dependency in service.dependencies:
            visit(dependency)
        visiting.remove(service_id)
        visited.add(service_id)

    for service_id in services:
        visit(service_id)
