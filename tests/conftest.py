"""Shared fixtures for all tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.stage3_fresh_database import STAGE3_TEST_DATABASE_NAME

load_dotenv(override=False)
_DEFAULT_CHARACTER_PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "personalities"
    / "kazusa.json"
)
os.environ.setdefault(
    "CHARACTER_PROFILE_PATH",
    str(_DEFAULT_CHARACTER_PROFILE_PATH),
)
if os.environ.get("MONGODB_DB_NAME") == STAGE3_TEST_DATABASE_NAME:
    os.environ["STAGE3_DATABASE_GUARD"] = "1"
else:
    os.environ["MONGODB_DB_NAME"] = "_test_kazusa_live_llm"
    os.environ.pop("STAGE3_DATABASE_GUARD", None)
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
