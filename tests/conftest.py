"""Shared fixtures for all tests."""

from __future__ import annotations

import asyncio
import os
import pytest
from dotenv import load_dotenv


collect_ignore_glob = ["*live_llm*.py"]
collect_ignore = [
    "test_coding_agent_phase_c_run_context_contracts.py",
    "test_self_cognition_response_sensitivity_helpers.py",
]


load_dotenv(override=False)
os.environ["MONGODB_DB_NAME"] = "_test_kazusa_live_llm"
os.environ["KAZUSA_TEST_DB_GUARD"] = "1"
os.environ.setdefault("CHARACTER_GLOBAL_USER_ID", "character-global")


# Disable langsmith in unit tests
@pytest.fixture(scope="session", autouse=True)
def disable_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""


@pytest.fixture(scope="session", autouse=True)
def close_mongodb_client():
    """Close the cached MongoDB client after the test session."""

    yield
    from kazusa_ai_chatbot.db._client import close_db

    asyncio.run(close_db())
