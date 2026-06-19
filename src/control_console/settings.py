"""Settings for the buildless local control console."""

from __future__ import annotations

from pathlib import Path
import os

from dotenv import dotenv_values, find_dotenv
from pydantic import BaseModel, ConfigDict, Field


class ControlConsoleSettings(BaseModel):
    """Runtime configuration for one local console process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    require_auth: bool = True
    operator_token_hash: str = ""
    local_only: bool = True
    app_instance_id: str = "local"
    session_cookie_name: str = "kazusa_control_session"
    csrf_header_name: str = "x-kazusa-control-csrf"
    state_dir: Path = Path(".kazusa_control")
    service_registry_path: Path | None = None
    brain_base_url: str = "http://127.0.0.1:8000"
    max_lookup_limit: int = Field(default=100, ge=1, le=500)
    max_log_lines: int = Field(default=500, ge=1, le=2000)
    max_event_limit: int = Field(default=200, ge=1, le=1000)
    sse_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    default_startup_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    default_shutdown_timeout_seconds: float = Field(default=15.0, gt=0, le=120)

    @classmethod
    def from_env(cls) -> "ControlConsoleSettings":
        """Build settings from `KAZUSA_CONTROL_*` environment variables."""

        dotenv_path = find_dotenv(usecwd=True)
        dotenv_config = dotenv_values(dotenv_path) if dotenv_path else {}
        registry_path = _control_setting(
            dotenv_config,
            "KAZUSA_CONTROL_SERVICE_REGISTRY",
            "",
        )
        state_dir = _control_setting(
            dotenv_config,
            "KAZUSA_CONTROL_STATE_DIR",
            ".kazusa_control",
        )
        settings = cls(
            host=_control_setting(
                dotenv_config,
                "KAZUSA_CONTROL_HOST",
                "127.0.0.1",
            ),
            port=int(_control_setting(dotenv_config, "KAZUSA_CONTROL_PORT", "8765")),
            operator_token_hash=_control_setting(
                dotenv_config,
                "KAZUSA_CONTROL_OPERATOR_TOKEN_HASH",
                "",
            ),
            state_dir=Path(state_dir),
            service_registry_path=Path(registry_path) if registry_path else None,
            brain_base_url=_control_setting(
                dotenv_config,
                "KAZUSA_CONTROL_BRAIN_BASE_URL",
                "http://127.0.0.1:8000",
            ),
        )
        return settings

    @property
    def audit_path(self) -> Path:
        """Return the local JSONL audit path."""

        audit_path = self.state_dir / "audit.jsonl"
        return audit_path

    @property
    def log_dir(self) -> Path:
        """Return the process-log directory."""

        log_dir = self.state_dir / "logs"
        return log_dir

    @property
    def process_state_dir(self) -> Path:
        """Return the process-state directory."""

        process_state_dir = self.state_dir
        return process_state_dir


def _control_setting(
    dotenv_config: dict[str, str | None],
    name: str,
    default: str,
) -> str:
    """Return a control-console setting with environment override precedence."""

    environment_value = os.environ.get(name)
    if environment_value is not None:
        return_value = environment_value
        return return_value

    dotenv_value = dotenv_config.get(name)
    if dotenv_value is not None:
        return_value = dotenv_value
        return return_value

    return_value = default
    return return_value
