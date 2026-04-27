"""Central configuration loaded from environment variables."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "roleplay_bot")

# LLM (LM Studio / OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")

# Optional secondary / specialized LLM routing.
# These default back to the primary model so deployments without a second model
# keep working unchanged.
SECONDARY_LLM_BASE_URL = os.getenv("SECONDARY_LLM_BASE_URL", LLM_BASE_URL)
SECONDARY_LLM_API_KEY = os.getenv("SECONDARY_LLM_API_KEY", LLM_API_KEY)
SECONDARY_LLM_MODEL = os.getenv("SECONDARY_LLM_MODEL", LLM_MODEL)

PREFERENCE_LLM_BASE_URL = os.getenv("PREFERENCE_LLM_BASE_URL", SECONDARY_LLM_BASE_URL)
PREFERENCE_LLM_API_KEY = os.getenv("PREFERENCE_LLM_API_KEY", SECONDARY_LLM_API_KEY)
PREFERENCE_LLM_MODEL = os.getenv("PREFERENCE_LLM_MODEL", SECONDARY_LLM_MODEL)

# Embedding model (LM Studio)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-model")

# Bot settings
CHARACTER_GLOBAL_USER_ID = os.getenv(
    "CHARACTER_GLOBAL_USER_ID",
    "00000000-0000-4000-8000-000000000001",
)
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10"))
# Recent history window for downstream stages.
CHAT_HISTORY_RECENT_LIMIT = 5

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
MAX_DIALOG_AGENT_RETRY = int(os.getenv("MAX_DIALOG_AGENT_RETRY", "3"))
MAX_FACT_HARVESTER_RETRY = int(os.getenv("MAX_FACT_HARVESTER_RETRY", "3"))

# RAG Cache2
RAG_CACHE2_MAX_ENTRIES = int(os.getenv("RAG_CACHE2_MAX_ENTRIES", "5000"))

# Persistent user-profile memory expiry defaults.
PROFILE_MEMORY_TTL_SECONDS = {
    "diary_entry": int(os.getenv("PROFILE_MEMORY_DIARY_TTL_SECONDS", str(90 * 24 * 60 * 60))),
    "objective_fact": int(os.getenv("PROFILE_MEMORY_FACT_TTL_SECONDS", str(365 * 24 * 60 * 60))),
    "milestone": int(os.getenv("PROFILE_MEMORY_MILESTONE_TTL_SECONDS", str(1095 * 24 * 60 * 60))),
    "commitment": int(os.getenv("PROFILE_MEMORY_COMMITMENT_TTL_SECONDS", str(10 * 24 * 60 * 60))),
}

PROFILE_MEMORY_RECENT_LIMITS = {
    "diary_entry": int(os.getenv("PROFILE_MEMORY_RECENT_DIARY_LIMIT", "6")),
    "objective_fact": int(os.getenv("PROFILE_MEMORY_RECENT_FACT_LIMIT", "8")),
    "milestone": int(os.getenv("PROFILE_MEMORY_RECENT_MILESTONE_LIMIT", "10")),
}

PROFILE_MEMORY_SEMANTIC_THRESHOLDS = {
    "diary_entry": float(os.getenv("PROFILE_MEMORY_DIARY_SEMANTIC_THRESHOLD", "0.75")),
    "objective_fact": float(os.getenv("PROFILE_MEMORY_FACT_SEMANTIC_THRESHOLD", "0.72")),
    "milestone": float(os.getenv("PROFILE_MEMORY_MILESTONE_SEMANTIC_THRESHOLD", "0.72")),
}

PROFILE_MEMORY_BUDGET = int(os.getenv("PROFILE_MEMORY_BUDGET", "40"))

# Scheduler (future_promise + followup_message events).
SCHEDULED_TASKS_ENABLED = os.getenv("SCHEDULED_TASKS_ENABLED", "true").lower() in ("1", "true", "yes")


# Brain service
BRAIN_EXECUTOR_COUNT = int(os.getenv("BRAIN_EXECUTOR_COUNT", "1"))
