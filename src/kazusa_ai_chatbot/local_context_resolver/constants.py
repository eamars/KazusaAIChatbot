"""Constants for the standalone local-context resolver."""

DEFAULT_OPTION_LIMITS = {
    "max_iterations": 3,
    "max_nodes": 8,
    "max_depth": 3,
    "max_node_attempts": 2,
    "max_subagent_attempts": 1,
}
OPTION_LIMIT_CAPS = {
    "max_iterations": 4,
    "max_nodes": 8,
    "max_depth": 3,
    "max_node_attempts": 2,
    "max_subagent_attempts": 1,
}
DEFAULT_SUBAGENT_MAX_ATTEMPTS = DEFAULT_OPTION_LIMITS["max_subagent_attempts"]

ROOT_NODE_ID = "root"
ROOT_NODE_DEPTH = 0
ROOT_CHILD_DEPTH = 1
STAGE_LLM_TEMPERATURE = 0.1
STAGE_LLM_TOP_P = 0.7
SAFE_FAILURE_TEXT_LIMIT = 300
TEXT_ELLIPSIS = "..."
