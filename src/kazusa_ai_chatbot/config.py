"""Central configuration loaded from environment variables."""

from __future__ import annotations

import json
import math
import os
from urllib.parse import urlparse

from dotenv import load_dotenv


def _positive_int_from_env(name: str, default: str) -> int:
    """Read a positive integer environment setting and fail fast if invalid."""

    raw_value = os.getenv(name, default)
    value = _positive_int_from_value(name, raw_value)
    return value


def _bounded_int_from_env(
    name: str,
    default: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Read an integer within inclusive bounds from the environment."""

    raw_value = os.getenv(name, default)
    value = int(raw_value)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _positive_int_from_value(name: str, raw_value: str) -> int:
    """Parse a positive integer config value and fail fast if invalid."""

    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _positive_int_from_env_alias(
    primary_name: str,
    legacy_name: str,
    default: str,
) -> int:
    """Read a positive integer with one temporary legacy env alias.

    Args:
        primary_name: Canonical environment variable name.
        legacy_name: Backward-compatible alias name.
        default: Default string used when neither variable is set.

    Returns:
        The configured positive integer.

    Raises:
        ValueError: If either value is invalid or both variables are set to
            different integer values.
    """

    primary_raw = os.getenv(primary_name)
    legacy_raw = os.getenv(legacy_name)
    if primary_raw is not None and legacy_raw is not None:
        primary_value = _positive_int_from_value(primary_name, primary_raw)
        legacy_value = _positive_int_from_value(legacy_name, legacy_raw)
        if primary_value != legacy_value:
            raise ValueError(
                f"{primary_name} conflicts with {legacy_name}"
            )
        return primary_value

    if primary_raw is not None:
        value = _positive_int_from_value(primary_name, primary_raw)
        return value

    if legacy_raw is not None:
        value = _positive_int_from_value(legacy_name, legacy_raw)
        return value

    value = _positive_int_from_value(primary_name, default)
    return value


def _bool_from_env_alias(
    primary_name: str,
    legacy_name: str,
    default: str,
) -> bool:
    """Read a bool with one temporary legacy env alias."""

    primary_raw = os.getenv(primary_name)
    legacy_raw = os.getenv(legacy_name)
    if primary_raw is not None and legacy_raw is not None:
        primary_value = _bool_from_value(primary_name, primary_raw)
        legacy_value = _bool_from_value(legacy_name, legacy_raw)
        if primary_value != legacy_value:
            raise ValueError(
                f"{primary_name} conflicts with {legacy_name}"
            )
        return primary_value

    if primary_raw is not None:
        value = _bool_from_value(primary_name, primary_raw)
        return value

    if legacy_raw is not None:
        value = _bool_from_value(legacy_name, legacy_raw)
        return value

    value = _bool_from_value(primary_name, default)
    return value


def _bounded_float_from_env(
    name: str,
    default: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite float within inclusive bounds from the environment."""

    raw_value = os.getenv(name, default)
    value = float(raw_value)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bool_from_env(name: str, default: str) -> bool:
    """Read a boolean environment setting and fail fast if invalid."""

    raw_value = os.getenv(name, default)
    value = _bool_from_value(name, raw_value)
    return value


def _bool_from_value(name: str, raw_value: str) -> bool:
    """Parse a boolean config value and fail fast if invalid."""

    normalized_value = raw_value.strip().lower()
    if normalized_value in ("1", "true", "yes"):
        return_value = True
        return return_value
    if normalized_value in ("0", "false", "no"):
        return_value = False
        return return_value
    raise ValueError(f"{name} must be a bool string")


def _non_empty_string_from_env(name: str, default: str) -> str:
    """Read a required non-empty string environment setting."""

    raw_value = os.getenv(name, default)
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _choice_from_env(name: str, default: str, allowed: set[str]) -> str:
    """Read a string choice from the environment and fail fast if invalid."""

    raw_value = os.getenv(name, default)
    value = raw_value.strip().lower()
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {allowed_text}")
    return value


def _optional_http_url_from_env(name: str, default: str) -> str:
    """Read an optional HTTP(S) URL, stripping whitespace and trailing slashes."""

    raw_value = os.getenv(name, default)
    value = raw_value.strip().rstrip("/")
    if not value:
        return ""

    parsed_url = urlparse(value)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError(f"{name} must be empty or an HTTP(S) URL")
    return value


def _local_time_minutes_from_value(name: str, value: str) -> int:
    """Parse exact ``HH:MM`` text into minutes after local midnight."""

    if len(value) != 5 or value[2] != ":":
        raise ValueError(f"{name} must use HH:MM-HH:MM")
    hour_text = value[:2]
    minute_text = value[3:]
    if not hour_text.isdecimal() or not minute_text.isdecimal():
        raise ValueError(f"{name} must use HH:MM-HH:MM")

    hour = int(hour_text)
    minute = int(minute_text)
    if hour > 23 or minute > 59:
        raise ValueError(f"{name} must use HH:MM-HH:MM")

    minutes = (hour * 60) + minute
    return minutes


def _optional_local_period_from_env(name: str, default: str) -> str:
    """Read an optional exact ``HH:MM-HH:MM`` local clock period."""

    raw_value = os.getenv(name, default)
    value = raw_value.strip()
    if not value:
        return ""

    parts = value.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"{name} must use HH:MM-HH:MM")
    start_minutes = _local_time_minutes_from_value(name, parts[0])
    end_minutes = _local_time_minutes_from_value(name, parts[1])
    if start_minutes == end_minutes:
        raise ValueError(f"{name} start and end must differ")

    return value


load_dotenv()

DEFAULT_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "DEFAULT_LLM_MAX_COMPLETION_TOKENS",
    "8192",
)

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "roleplay_bot")

# Route-specific chat LLMs. These are intentionally required: a missing route
# variable means the deployment configuration is incomplete.
RELEVANCE_AGENT_LLM_BASE_URL = os.environ["RELEVANCE_AGENT_LLM_BASE_URL"]
RELEVANCE_AGENT_LLM_API_KEY = os.environ["RELEVANCE_AGENT_LLM_API_KEY"]
RELEVANCE_AGENT_LLM_MODEL = os.environ["RELEVANCE_AGENT_LLM_MODEL"]

VISION_DESCRIPTOR_LLM_BASE_URL = os.environ["VISION_DESCRIPTOR_LLM_BASE_URL"]
VISION_DESCRIPTOR_LLM_API_KEY = os.environ["VISION_DESCRIPTOR_LLM_API_KEY"]
VISION_DESCRIPTOR_LLM_MODEL = os.environ["VISION_DESCRIPTOR_LLM_MODEL"]

MSG_DECONTEXTUALIZER_LLM_BASE_URL = os.environ["MSG_DECONTEXTUALIZER_LLM_BASE_URL"]
MSG_DECONTEXTUALIZER_LLM_API_KEY = os.environ["MSG_DECONTEXTUALIZER_LLM_API_KEY"]
MSG_DECONTEXTUALIZER_LLM_MODEL = os.environ["MSG_DECONTEXTUALIZER_LLM_MODEL"]

RAG_PLANNER_LLM_BASE_URL = os.environ["RAG_PLANNER_LLM_BASE_URL"]
RAG_PLANNER_LLM_API_KEY = os.environ["RAG_PLANNER_LLM_API_KEY"]
RAG_PLANNER_LLM_MODEL = os.environ["RAG_PLANNER_LLM_MODEL"]

RAG_SUBAGENT_LLM_BASE_URL = os.environ["RAG_SUBAGENT_LLM_BASE_URL"]
RAG_SUBAGENT_LLM_API_KEY = os.environ["RAG_SUBAGENT_LLM_API_KEY"]
RAG_SUBAGENT_LLM_MODEL = os.environ["RAG_SUBAGENT_LLM_MODEL"]

WEB_SEARCH_LLM_BASE_URL = os.environ["WEB_SEARCH_LLM_BASE_URL"]
WEB_SEARCH_LLM_API_KEY = os.environ["WEB_SEARCH_LLM_API_KEY"]
WEB_SEARCH_LLM_MODEL = os.environ["WEB_SEARCH_LLM_MODEL"]

COGNITION_LLM_BASE_URL = os.environ["COGNITION_LLM_BASE_URL"]
COGNITION_LLM_API_KEY = os.environ["COGNITION_LLM_API_KEY"]
COGNITION_LLM_MODEL = os.environ["COGNITION_LLM_MODEL"]

BOUNDARY_CORE_LLM_BASE_URL = os.environ["BOUNDARY_CORE_LLM_BASE_URL"]
BOUNDARY_CORE_LLM_API_KEY = os.environ["BOUNDARY_CORE_LLM_API_KEY"]
BOUNDARY_CORE_LLM_MODEL = os.environ["BOUNDARY_CORE_LLM_MODEL"]

DIALOG_GENERATOR_LLM_BASE_URL = os.environ["DIALOG_GENERATOR_LLM_BASE_URL"]
DIALOG_GENERATOR_LLM_API_KEY = os.environ["DIALOG_GENERATOR_LLM_API_KEY"]
DIALOG_GENERATOR_LLM_MODEL = os.environ["DIALOG_GENERATOR_LLM_MODEL"]

CONSOLIDATION_LLM_BASE_URL = os.environ["CONSOLIDATION_LLM_BASE_URL"]
CONSOLIDATION_LLM_API_KEY = os.environ["CONSOLIDATION_LLM_API_KEY"]
CONSOLIDATION_LLM_MODEL = os.environ["CONSOLIDATION_LLM_MODEL"]

JSON_REPAIR_LLM_BASE_URL = os.environ["JSON_REPAIR_LLM_BASE_URL"]
JSON_REPAIR_LLM_API_KEY = os.environ["JSON_REPAIR_LLM_API_KEY"]
JSON_REPAIR_LLM_MODEL = os.environ["JSON_REPAIR_LLM_MODEL"]

BACKGROUND_ARTIFACT_LLM_BASE_URL = os.getenv(
    "BACKGROUND_ARTIFACT_LLM_BASE_URL",
    COGNITION_LLM_BASE_URL,
)
BACKGROUND_ARTIFACT_LLM_API_KEY = os.getenv(
    "BACKGROUND_ARTIFACT_LLM_API_KEY",
    COGNITION_LLM_API_KEY,
)
BACKGROUND_ARTIFACT_LLM_MODEL = os.getenv(
    "BACKGROUND_ARTIFACT_LLM_MODEL",
    COGNITION_LLM_MODEL,
)
BACKGROUND_WORK_LLM_BASE_URL = os.getenv(
    "BACKGROUND_WORK_LLM_BASE_URL",
    BACKGROUND_ARTIFACT_LLM_BASE_URL,
)
BACKGROUND_WORK_LLM_API_KEY = os.getenv(
    "BACKGROUND_WORK_LLM_API_KEY",
    BACKGROUND_ARTIFACT_LLM_API_KEY,
)
BACKGROUND_WORK_LLM_MODEL = os.getenv(
    "BACKGROUND_WORK_LLM_MODEL",
    BACKGROUND_ARTIFACT_LLM_MODEL,
)

CODING_AGENT_PM_LLM_BASE_URL = os.environ["CODING_AGENT_PM_LLM_BASE_URL"]
CODING_AGENT_PM_LLM_API_KEY = os.environ["CODING_AGENT_PM_LLM_API_KEY"]
CODING_AGENT_PM_LLM_MODEL = os.environ["CODING_AGENT_PM_LLM_MODEL"]

CODING_AGENT_PROGRAMMER_LLM_BASE_URL = os.environ[
    "CODING_AGENT_PROGRAMMER_LLM_BASE_URL"
]
CODING_AGENT_PROGRAMMER_LLM_API_KEY = os.environ[
    "CODING_AGENT_PROGRAMMER_LLM_API_KEY"
]
CODING_AGENT_PROGRAMMER_LLM_MODEL = os.environ[
    "CODING_AGENT_PROGRAMMER_LLM_MODEL"
]

RELEVANCE_AGENT_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "RELEVANCE_AGENT_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
RELEVANCE_AGENT_LLM_THINKING_ENABLED = _bool_from_env(
    "RELEVANCE_AGENT_LLM_THINKING_ENABLED",
    "false",
)
VISION_DESCRIPTOR_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "VISION_DESCRIPTOR_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
VISION_DESCRIPTOR_LLM_THINKING_ENABLED = _bool_from_env(
    "VISION_DESCRIPTOR_LLM_THINKING_ENABLED",
    "false",
)
MSG_DECONTEXTUALIZER_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "MSG_DECONTEXTUALIZER_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
MSG_DECONTEXTUALIZER_LLM_THINKING_ENABLED = _bool_from_env(
    "MSG_DECONTEXTUALIZER_LLM_THINKING_ENABLED",
    "false",
)
RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
RAG_PLANNER_LLM_THINKING_ENABLED = _bool_from_env(
    "RAG_PLANNER_LLM_THINKING_ENABLED",
    "false",
)
RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
RAG_SUBAGENT_LLM_THINKING_ENABLED = _bool_from_env(
    "RAG_SUBAGENT_LLM_THINKING_ENABLED",
    "false",
)
WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
WEB_SEARCH_LLM_THINKING_ENABLED = _bool_from_env(
    "WEB_SEARCH_LLM_THINKING_ENABLED",
    "false",
)
COGNITION_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "COGNITION_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
COGNITION_LLM_THINKING_ENABLED = _bool_from_env(
    "COGNITION_LLM_THINKING_ENABLED",
    "false",
)
BOUNDARY_CORE_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "BOUNDARY_CORE_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
BOUNDARY_CORE_LLM_THINKING_ENABLED = _bool_from_env(
    "BOUNDARY_CORE_LLM_THINKING_ENABLED",
    "false",
)
DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
DIALOG_GENERATOR_LLM_THINKING_ENABLED = _bool_from_env(
    "DIALOG_GENERATOR_LLM_THINKING_ENABLED",
    "false",
)
CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
CONSOLIDATION_LLM_THINKING_ENABLED = _bool_from_env(
    "CONSOLIDATION_LLM_THINKING_ENABLED",
    "false",
)
JSON_REPAIR_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "JSON_REPAIR_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
JSON_REPAIR_LLM_THINKING_ENABLED = _bool_from_env(
    "JSON_REPAIR_LLM_THINKING_ENABLED",
    "false",
)
BACKGROUND_ARTIFACT_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "BACKGROUND_ARTIFACT_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
BACKGROUND_ARTIFACT_LLM_THINKING_ENABLED = _bool_from_env(
    "BACKGROUND_ARTIFACT_LLM_THINKING_ENABLED",
    "false",
)
BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
BACKGROUND_WORK_LLM_THINKING_ENABLED = _bool_from_env(
    "BACKGROUND_WORK_LLM_THINKING_ENABLED",
    "false",
)
CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
CODING_AGENT_PM_LLM_THINKING_ENABLED = _bool_from_env(
    "CODING_AGENT_PM_LLM_THINKING_ENABLED",
    "false",
)
CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS = _positive_int_from_env(
    "CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS",
    str(DEFAULT_LLM_MAX_COMPLETION_TOKENS),
)
CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED = _bool_from_env(
    "CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED",
    "false",
)

# Embedding model (LM Studio)
EMBEDDING_BASE_URL = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_API_KEY = os.environ["EMBEDDING_API_KEY"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# Bot settings
CHARACTER_GLOBAL_USER_ID = _non_empty_string_from_env(
    "CHARACTER_GLOBAL_USER_ID",
    "00000000-0000-4000-8000-000000000001",
)
AUDIT_LOG_TTL_DAYS = _positive_int_from_env("AUDIT_LOG_TTL_DAYS", "90")
DEBUG_LOG_TTL_DAYS = _positive_int_from_env("DEBUG_LOG_TTL_DAYS", "14")
LLM_TRACE_CAPTURE_MODE = _choice_from_env(
    "LLM_TRACE_CAPTURE_MODE",
    "metadata",
    {"off", "metadata", "full"},
)
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10"))
RAG_SEARCH_DEFAULT_TOP_K = _positive_int_from_env_alias(
    "RAG_SEARCH_DEFAULT_TOP_K",
    "CONVERSATION_SEARCH_DEFAULT_TOP_K",
    "20",
)
RAG_SEARCH_MAX_TOP_K = _positive_int_from_env_alias(
    "RAG_SEARCH_MAX_TOP_K",
    "CONVERSATION_SEARCH_MAX_TOP_K",
    "50",
)
if RAG_SEARCH_MAX_TOP_K < RAG_SEARCH_DEFAULT_TOP_K:
    raise ValueError(
        "RAG_SEARCH_MAX_TOP_K must be >= "
        "RAG_SEARCH_DEFAULT_TOP_K"
    )
RAG_SEARCH_SELECTED_LIMIT = _positive_int_from_env(
    "RAG_SEARCH_SELECTED_LIMIT",
    "20",
)
RAG_SEARCH_SELECTED_SUMMARY_LIMIT = _positive_int_from_env(
    "RAG_SEARCH_SELECTED_SUMMARY_LIMIT",
    "20",
)
RAG_VECTOR_MIN_CANDIDATES = _positive_int_from_env(
    "RAG_VECTOR_MIN_CANDIDATES",
    "200",
)
RAG_VECTOR_CANDIDATE_MULTIPLIER = _positive_int_from_env(
    "RAG_VECTOR_CANDIDATE_MULTIPLIER",
    "20",
)
RAG_VECTOR_MAX_CANDIDATES = _positive_int_from_env(
    "RAG_VECTOR_MAX_CANDIDATES",
    "10000",
)
RAG_HYBRID_NEIGHBOR_SEED_LIMIT = _positive_int_from_env(
    "RAG_HYBRID_NEIGHBOR_SEED_LIMIT",
    "8",
)
RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT = _positive_int_from_env(
    "RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT",
    "3",
)
RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES = _positive_int_from_env(
    "RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES",
    "3",
)
RAG_HYBRID_LITERAL_ANCHOR_LIMIT = _positive_int_from_env(
    "RAG_HYBRID_LITERAL_ANCHOR_LIMIT",
    "5",
)
RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR = _bounded_float_from_env(
    "RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR",
    "0.72",
    minimum=0.0,
    maximum=1.0,
)
RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT = _positive_int_from_env(
    "RAG_CONVERSATION_EVIDENCE_TEXT_LIMIT",
    "500",
)
RAG_MEMORY_EVIDENCE_TEXT_LIMIT = _positive_int_from_env(
    "RAG_MEMORY_EVIDENCE_TEXT_LIMIT",
    "500",
)
CONVERSATION_SEARCH_DEFAULT_TOP_K = RAG_SEARCH_DEFAULT_TOP_K
CONVERSATION_SEARCH_MAX_TOP_K = RAG_SEARCH_MAX_TOP_K
SAVE_ATTACHMENT_BASE64_TO_DB = os.getenv(
    "SAVE_ATTACHMENT_BASE64_TO_DB",
    "false",
).lower() in ("1", "true", "yes")
# Recent history window for downstream stages.
CHAT_HISTORY_RECENT_LIMIT = 5

# Direct web search and URL reader settings.
SEARXNG_URL = _optional_http_url_from_env("SEARXNG_URL", "")
SEARXNG_SEARCH_TIMEOUT_SECONDS = _bounded_float_from_env(
    "SEARXNG_SEARCH_TIMEOUT_SECONDS",
    "30",
    minimum=1.0,
    maximum=120.0,
)
SEARXNG_SEARCH_RESULT_LIMIT = _bounded_int_from_env(
    "SEARXNG_SEARCH_RESULT_LIMIT",
    "10",
    minimum=1,
    maximum=20,
)
WEB_URL_READ_TIMEOUT_SECONDS = _bounded_float_from_env(
    "WEB_URL_READ_TIMEOUT_SECONDS",
    "30",
    minimum=1.0,
    maximum=120.0,
)
WEB_URL_READ_MAX_BYTES = _bounded_int_from_env(
    "WEB_URL_READ_MAX_BYTES",
    "1048576",
    minimum=1024,
    maximum=5242880,
)
WEB_URL_READ_MAX_CHARS = _bounded_int_from_env(
    "WEB_URL_READ_MAX_CHARS",
    "10000",
    minimum=1000,
    maximum=50000,
)
WEB_URL_READ_REDIRECT_LIMIT = _bounded_int_from_env(
    "WEB_URL_READ_REDIRECT_LIMIT",
    "5",
    minimum=0,
    maximum=10,
)
WEB_URL_READER_USER_AGENT = _non_empty_string_from_env(
    "WEB_URL_READER_USER_AGENT",
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
)
WEB_URL_READER_ACCEPT_LANGUAGE = _non_empty_string_from_env(
    "WEB_URL_READER_ACCEPT_LANGUAGE",
    "en-US,en;q=0.9",
)

# Maximum guideline strings retained for each interaction-style category in
# persisted user or group-channel style images.
INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT = _positive_int_from_env(
    "INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT",
    "5",
)
# Maximum stored guideline strings projected from each style source into L3
# cognition prompts. Group chat may include one user source and one group source.
L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT = _positive_int_from_env(
    "L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT",
    "5",
)
# Maximum stored user engagement guidelines projected into the group relevance
# prompt. This keeps relevance context bounded on the live response path.
RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT = _positive_int_from_env(
    "RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT",
    "3",
)

INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE = _bounded_int_from_env(
    "INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE",
    "5",
    minimum=1,
    maximum=10,
)
INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT = _bounded_int_from_env(
    "INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT",
    "3000",
    minimum=200,
    maximum=3000,
)
INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT = _bounded_int_from_env(
    "INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT",
    "220",
    minimum=80,
    maximum=500,
)

# MCP tool servers — JSON dict of {name: {url: ...}}
# Read from MCP_SERVERS env var (JSON string)
mcp_servers_env = os.getenv("MCP_SERVERS", "{}")
# Handle escaped quotes that might come from shell environment
mcp_servers_env = mcp_servers_env.replace('\\"', '"')
MCP_SERVERS: dict[str, dict] = json.loads(mcp_servers_env)

# Seconds to wait for a single tool call before giving up.
MCP_CALL_TIMEOUT: float = float(os.getenv("MCP_CALL_TIMEOUT", "30"))
# Seconds to wait for server initialisation and tool-list discovery at startup.
MCP_CONNECT_TIMEOUT: float = float(os.getenv("MCP_CONNECT_TIMEOUT", "10"))

# Affinity system
# Affinity scaling breakpoints (later should be read from character profile)
AFFINITY_INCREMENT_BREAKPOINTS = [
    (0, 1.5),      # At 0: 1.5x scaling (easy to gain)
    (300, 1.5),   # At 300: 1.5x scaling (still easy)
    (300, 1.0),   # At 300: 1.0x scaling (normal starts)
    (700, 1.0),   # At 700: 1.0x scaling (normal ends)
    (700, 0.6),   # At 700: 0.6x scaling (harder to gain)
    (1000, 0.6)   # At 1000: 0.6x scaling (hardest to gain)
]

AFFINITY_DECREMENT_BREAKPOINTS = [
    (0, 1.3),      # At 0: 1.3x scaling (easy to lose when very low)
    (300, 1.3),   # At 300: 1.3x scaling (still easy to lose)
    (300, 1.0),   # At 300: 1.0x scaling (normal starts)
    (700, 1.0),   # At 700: 1.0x scaling (normal ends)
    (700, 0.6),   # At 700: 0.6x scaling (harder to lose)
    (1000, 0.6)   # At 1000: 0.6x scaling (hardest to lose)
]
AFFINITY_DEFAULT = 500
AFFINITY_MIN = 0
AFFINITY_MAX = 1000
AFFINITY_RAW_DEAD_ZONE = int(os.getenv("AFFINITY_RAW_DEAD_ZONE", "1"))

# Loop counts
MAX_MEMORY_RETRIEVER_AGENT_RETRY = int(os.getenv("MAX_MEMORY_RETRIEVER_AGENT_RETRY", "2"))
MAX_WEB_SEARCH_AGENT_RETRY = int(os.getenv("MAX_WEB_SEARCH_AGENT_RETRY", "2"))
MAX_FACT_HARVESTER_RETRY = int(os.getenv("MAX_FACT_HARVESTER_RETRY", "3"))

# RAG Cache2
RAG_CACHE2_MAX_ENTRIES = int(os.getenv("RAG_CACHE2_MAX_ENTRIES", "5000"))

# Media descriptor persistent cache
MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES = int(
    os.getenv("MEDIA_DESCRIPTOR_CACHE_MAX_PERSISTENT_ENTRIES", "500")
)
MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES = int(
    os.getenv("MEDIA_DESCRIPTOR_CACHE_MAX_HYDRATION_ENTRIES", "100")
)

# Calendar scheduler durable worker settings.
CALENDAR_SCHEDULER_ENABLED = _bool_from_env(
    "CALENDAR_SCHEDULER_ENABLED",
    "true",
)
CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS = _positive_int_from_env(
    "CALENDAR_SCHEDULER_POLL_INTERVAL_SECONDS",
    "30",
)
CALENDAR_SCHEDULER_CLAIM_LIMIT = _positive_int_from_env(
    "CALENDAR_SCHEDULER_CLAIM_LIMIT",
    "10",
)
CALENDAR_SCHEDULER_LEASE_SECONDS = _positive_int_from_env(
    "CALENDAR_SCHEDULER_LEASE_SECONDS",
    "300",
)
CALENDAR_SCHEDULER_MAX_ATTEMPTS = _positive_int_from_env(
    "CALENDAR_SCHEDULER_MAX_ATTEMPTS",
    "3",
)
CALENDAR_SCHEDULER_PER_TRIGGER_CAPACITY = _positive_int_from_env(
    "CALENDAR_SCHEDULER_PER_TRIGGER_CAPACITY",
    "5",
)

# Visual directives are service-side generation metadata, not an adapter debug
# mode. Disable this to skip the L3 visual-agent LLM call globally.
COGNITION_VISUAL_DIRECTIVES_ENABLED = os.getenv(
    "COGNITION_VISUAL_DIRECTIVES_ENABLED",
    "true",
).lower() in ("1", "true", "yes")

COGNITION_RESOLVER_MAX_CYCLES = _bounded_int_from_env(
    "COGNITION_RESOLVER_MAX_CYCLES",
    "3",
    minimum=1,
    maximum=5,
)
COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS = _bounded_float_from_env(
    "COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS",
    "120.0",
    minimum=1.0,
    maximum=180.0,
)

# Self-cognition runs by default as a background reasoning loop. Its runtime
# output and persistence boundaries stay controlled by the self-cognition worker.
SELF_COGNITION_ENABLED = _bool_from_env("SELF_COGNITION_ENABLED", "true")
SELF_COGNITION_WORKER_INTERVAL_SECONDS = _positive_int_from_env(
    "SELF_COGNITION_WORKER_INTERVAL_SECONDS",
    "3600",
)
SELF_COGNITION_MAX_CASES_PER_TICK = _positive_int_from_env(
    "SELF_COGNITION_MAX_CASES_PER_TICK",
    "3",
)

# Background work runs as asynchronous worker-subagent jobs. Legacy
# BACKGROUND_ARTIFACT_* settings remain aliases for existing deployments.
BACKGROUND_WORK_WORKER_ENABLED = _bool_from_env_alias(
    "BACKGROUND_WORK_WORKER_ENABLED",
    "BACKGROUND_ARTIFACT_WORKER_ENABLED",
    "true",
)
BACKGROUND_WORK_WORKER_INTERVAL_SECONDS = _positive_int_from_env_alias(
    "BACKGROUND_WORK_WORKER_INTERVAL_SECONDS",
    "BACKGROUND_ARTIFACT_WORKER_INTERVAL_SECONDS",
    "45",
)
BACKGROUND_WORK_WORKER_CLAIM_LIMIT = _positive_int_from_env_alias(
    "BACKGROUND_WORK_WORKER_CLAIM_LIMIT",
    "BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT",
    "2",
)
BACKGROUND_WORK_WORKER_LEASE_SECONDS = _positive_int_from_env_alias(
    "BACKGROUND_WORK_WORKER_LEASE_SECONDS",
    "BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS",
    "180",
)
BACKGROUND_WORK_WORKER_MAX_ATTEMPTS = _positive_int_from_env_alias(
    "BACKGROUND_WORK_WORKER_MAX_ATTEMPTS",
    "BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS",
    "4",
)
BACKGROUND_WORK_INPUT_CHAR_LIMIT = _positive_int_from_env_alias(
    "BACKGROUND_WORK_INPUT_CHAR_LIMIT",
    "BACKGROUND_ARTIFACT_INPUT_CHAR_LIMIT",
    "8000",
)
BACKGROUND_WORK_OUTPUT_CHAR_LIMIT = _positive_int_from_env_alias(
    "BACKGROUND_WORK_OUTPUT_CHAR_LIMIT",
    "BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT",
    "3000",
)
BACKGROUND_WORK_DELIVERY_MAX_ATTEMPTS = _positive_int_from_env(
    "BACKGROUND_WORK_DELIVERY_MAX_ATTEMPTS",
    "6",
)
BACKGROUND_ARTIFACT_WORKER_ENABLED = BACKGROUND_WORK_WORKER_ENABLED
BACKGROUND_ARTIFACT_WORKER_INTERVAL_SECONDS = (
    BACKGROUND_WORK_WORKER_INTERVAL_SECONDS
)
BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT = BACKGROUND_WORK_WORKER_CLAIM_LIMIT
BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS = BACKGROUND_WORK_WORKER_LEASE_SECONDS
BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS = BACKGROUND_WORK_WORKER_MAX_ATTEMPTS
BACKGROUND_ARTIFACT_INPUT_CHAR_LIMIT = BACKGROUND_WORK_INPUT_CHAR_LIMIT
BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT
# Source packets enter cognition as internal-monologue percepts, so the default
# budget stays aligned with the existing internal-thought cognition boundary.
SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT = _positive_int_from_env(
    "SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT",
    "4000",
)
SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED = _bool_from_env(
    "SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED",
    "true",
)
SELF_COGNITION_TRIGGER_CONVERSATION_PROGRESS_ENABLED = _bool_from_env(
    "SELF_COGNITION_TRIGGER_CONVERSATION_PROGRESS_ENABLED",
    "true",
)
SELF_COGNITION_TRIGGER_RECENT_DIRECT_DIALOG_ENABLED = _bool_from_env(
    "SELF_COGNITION_TRIGGER_RECENT_DIRECT_DIALOG_ENABLED",
    "true",
)
SELF_COGNITION_TRIGGER_PENDING_OUTBOX_ENABLED = _bool_from_env(
    "SELF_COGNITION_TRIGGER_PENDING_OUTBOX_ENABLED",
    "true",
)
SELF_COGNITION_TRIGGER_BOUNDED_TOPIC_FOLLOWUP_ENABLED = _bool_from_env(
    "SELF_COGNITION_TRIGGER_BOUNDED_TOPIC_FOLLOWUP_ENABLED",
    "true",
)
SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED = _bool_from_env(
    "SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED",
    "true",
)
CHARACTER_SLEEP_LOCAL_PERIOD = _optional_local_period_from_env(
    "CHARACTER_SLEEP_LOCAL_PERIOD",
    "02:00-12:00",
)

# Character timezone (IANA name) for converting UTC to character-local time.
CHARACTER_TIME_ZONE = os.getenv("CHARACTER_TIME_ZONE", "Pacific/Auckland")

# Reflection cycle
REFLECTION_CYCLE_ENABLED = os.getenv(
    "REFLECTION_CYCLE_ENABLED",
    "true",
).lower() in ("1", "true", "yes")
REFLECTION_LORE_PROMOTION_ENABLED = os.getenv(
    "REFLECTION_LORE_PROMOTION_ENABLED",
    "true",
).lower() in ("1", "true", "yes")
REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED = os.getenv(
    "REFLECTION_SELF_GUIDANCE_PROMOTION_ENABLED",
    "true",
).lower() in ("1", "true", "yes")
REFLECTION_WORKER_INTERVAL_SECONDS = int(
    os.getenv("REFLECTION_WORKER_INTERVAL_SECONDS", "900")
)
REFLECTION_HOURLY_SLOTS_PER_TICK = int(
    os.getenv("REFLECTION_HOURLY_SLOTS_PER_TICK", "3")
)
REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS = _positive_int_from_env(
    "REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS",
    "60",
)
REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD = _positive_int_from_env(
    "REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD",
    str(REFLECTION_HOURLY_SLOTS_PER_TICK),
)
_allowed_reflection_phase_slots = (
    (REFLECTION_WORKER_INTERVAL_SECONDS - 1)
    // REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS
) + 1
if REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD > _allowed_reflection_phase_slots:
    raise ValueError(
        "REFLECTION_PHASE_MAX_SLOTS_PER_PERIOD cannot fit inside "
        "REFLECTION_WORKER_INTERVAL_SECONDS with "
        "REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS"
    )
REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME = os.getenv(
    "REFLECTION_DAILY_RUN_AFTER_LOCAL_TIME",
    "04:30",
)
REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME = os.getenv(
    "REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME",
    "05:00",
)
GLOBAL_CHARACTER_GROWTH_PASS_ENABLED = os.getenv(
    "GLOBAL_CHARACTER_GROWTH_PASS_ENABLED",
    "true",
).lower() in ("1", "true", "yes")
GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET = _positive_int_from_env(
    "GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET",
    "32000",
)

# Brain service
BRAIN_EXECUTOR_COUNT = int(os.getenv("BRAIN_EXECUTOR_COUNT", "1"))
