"""Central logging configuration for the Kazusa chatbot."""

from __future__ import annotations

import logging


APP_LOG_LEVEL = logging.DEBUG
THIRD_PARTY_LOG_LEVEL = logging.WARNING
SERVICE_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
ADAPTER_LOG_FORMAT = "%(asctime)s [%(levelname)s] adapter:%(name)s: %(message)s"
QUIET_LOGGERS = (
    "asyncio",
    "discord",
    "hpack",
    "httpcore",
    "httpx",
    "langsmith",
    "mcp",
    "mcp.client",
    "mcp.client.streamable_http",
    "openai",
    "pymongo",
    "urllib3",
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
)

_configured_profile: str | None = None


def configure_service_logging() -> None:
    """Apply the central brain-service logging policy.

    Returns:
        None. The root logger is configured with one stream handler, application
        loggers inherit DEBUG visibility, and noisy dependencies are raised to
        WARNING so caller code does not decide logging behavior.
    """
    _configure_logging(profile="service", log_format=SERVICE_LOG_FORMAT)


def configure_adapter_logging() -> None:
    """Apply the central adapter logging policy.

    Returns:
        None. Adapter processes use a separate root format from the brain
        service while keeping the same no-hidden-debug-information level.
    """
    _configure_logging(profile="adapter", log_format=ADAPTER_LOG_FORMAT)


def configure_logging() -> None:
    """Apply the default central logging policy.

    Returns:
        None. This compatibility wrapper keeps package imports and scripts on
        the service policy unless a runtime explicitly selects another policy.
    """
    configure_service_logging()


def _configure_logging(*, profile: str, log_format: str) -> None:
    """Install one central logging profile.

    Args:
        profile: Stable name for the current runtime logging profile.
        log_format: Root handler format for the runtime.

    Returns:
        None.
    """
    global _configured_profile
    if _configured_profile == profile:
        return

    logging.basicConfig(
        level=APP_LOG_LEVEL,
        format=log_format,
        force=True,
    )
    for logger_name in QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(THIRD_PARTY_LOG_LEVEL)
    _configured_profile = profile
