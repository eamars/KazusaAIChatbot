"""Shared fixtures for all tests."""

from __future__ import annotations

import os
import pytest
from dotenv import load_dotenv


load_dotenv(override=False)
os.environ.setdefault("CHARACTER_GLOBAL_USER_ID", "character-global")


# Disable langsmith in unit tests
@pytest.fixture(scope="session", autouse=True)
def disable_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""
