import json
import os

from dotenv import load_dotenv

load_dotenv()

# Discord
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "roleplay_bot")

# LLM (LM Studio / OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.getenv("LLM_MODEL", "local-model")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Embedding model (LM Studio)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-model")

# Bot settings
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "28000"))

# Token budget allocation (approximate)
TOKEN_BUDGET = {
    "system_personality": 15000,
    "character_state": 500,
    "user_memory": 500,
    "conversation_history": 4000,
    "current_message": 500,
}

# MCP tool servers — JSON dict of {name: {url: ...}}
# Read from MCP_SERVERS env var (JSON string)
mcp_servers_env = os.getenv("MCP_SERVERS", "{}")
# Handle escaped quotes that might come from shell environment
mcp_servers_env = mcp_servers_env.replace('\\"', '"')
MCP_SERVERS: dict[str, dict] = json.loads(mcp_servers_env)

# Max tool-calling loop iterations in persona supervisor
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "3"))

# Max supervisor evaluate-dispatch loop iterations
MAX_SUPERVISOR_ITERATIONS = int(os.getenv("MAX_SUPERVISOR_ITERATIONS", "3"))

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


# Loop counts
MAX_RESEARCH_AGENT_RETRY = int(os.getenv("MAX_RESEARCH_AGENT_RETRY", "2"))
MAX_MEMORY_RETRIEVER_AGENT_RETRY = int(os.getenv("MAX_MEMORY_RETRIEVER_AGENT_RETRY", "2"))
MAX_WEB_SEARCH_AGENT_RETRY = int(os.getenv("MAX_WEB_SEARCH_AGENT_RETRY", "2"))
MAX_DIALOG_AGENT_RETRY = int(os.getenv("MAX_DIALOG_AGENT_RETRY", "3"))
MAX_FACT_HARVESTER_RETRY = int(os.getenv("MAX_FACT_HARVESTER_RETRY", "3"))

# ── Stage 5a: RAG cache, depth classifier, consolidator, scheduler ──
# RAG cache (semantic LRU + MongoDB write-through). TTLs override the
# per-type defaults inside ``rag.cache.DEFAULT_TTL_SECONDS`` when the cache
# is constructed via ``build_rag_cache()``.
RAG_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("RAG_CACHE_SIMILARITY_THRESHOLD", "0.65"))
RAG_CACHE_MAX_SIZE = int(os.getenv("RAG_CACHE_MAX_SIZE", "100000"))
RAG_CACHE_TTL_SECONDS = {
    "character_diary": int(os.getenv("RAG_CACHE_CHARACTER_DIARY_TTL", "1800")),
    "objective_user_facts": int(os.getenv("RAG_CACHE_OBJECTIVE_USER_FACTS_TTL", "3600")),
    "user_promises": int(os.getenv("RAG_CACHE_USER_PROMISES_TTL", "900")),
    "internal_memory": int(os.getenv("RAG_CACHE_INTERNAL_MEMORY_TTL", "900")),
    "external_knowledge": int(os.getenv("RAG_CACHE_EXTERNAL_KNOWLEDGE_TTL", "3600")),
    "knowledge_base": int(os.getenv("RAG_CACHE_KNOWLEDGE_BASE_TTL", "2592000")),  # 30 days
}

# Depth classifier (SHALLOW vs DEEP routing for the RAG dispatcher).
DEPTH_CLASSIFIER_USE_LIGHT_LLM = os.getenv("DEPTH_CLASSIFIER_USE_LIGHT_LLM", "false").lower() in ("1", "true", "yes")
DEPTH_CLASSIFIER_THRESHOLDS = {
    "shallow_max_chars": int(os.getenv("DEPTH_SHALLOW_MAX_CHARS", "60")),
    "embedding_confidence_min": float(os.getenv("DEPTH_EMBEDDING_CONFIDENCE_MIN", "0.55")),
}

# Consolidator (Stage 4a evaluator + cache invalidation).
EVALUATOR_CONSISTENCY_CHECK = os.getenv("EVALUATOR_CONSISTENCY_CHECK", "true").lower() in ("1", "true", "yes")
AFFINITY_CACHE_NUKE_THRESHOLD = int(os.getenv("AFFINITY_CACHE_NUKE_THRESHOLD", "50"))

# Scheduler (future_promise + followup_message events).
SCHEDULED_TASKS_ENABLED = os.getenv("SCHEDULED_TASKS_ENABLED", "true").lower() in ("1", "true", "yes")


# Brain service
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8000"))
BRAIN_EXECUTOR_COUNT = int(os.getenv("BRAIN_EXECUTOR_COUNT", "1"))