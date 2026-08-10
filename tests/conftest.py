"""Shared fixtures for all tests."""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from dotenv import load_dotenv

from tests.stage3_fresh_database import STAGE3_TEST_DATABASE_NAME


def pytest_collection_modifyitems(config, items):
    """Require every explicitly selected impact node to be collected."""

    del config
    if os.environ.get("KAZUSA_TEST_IMPACT_REQUIRED") != "1":
        return
    required_nodes_text = os.environ.get("KAZUSA_TEST_IMPACT_NODES")
    if not required_nodes_text:
        raise pytest.UsageError(
            "KAZUSA_TEST_IMPACT_NODES is required when impact enforcement is on"
        )
    try:
        required_nodes = json.loads(required_nodes_text)
    except json.JSONDecodeError as exc:
        raise pytest.UsageError(
            f"KAZUSA_TEST_IMPACT_NODES is invalid JSON: {exc}"
        ) from exc
    if not isinstance(required_nodes, list) or not all(
        isinstance(node_id, str) for node_id in required_nodes
    ):
        raise pytest.UsageError(
            "KAZUSA_TEST_IMPACT_NODES must be a JSON list of node IDs"
        )
    collected_nodes = {
        item.nodeid.replace("\\", "/")
        for item in items
    }
    missing_nodes = sorted(
        node_id
        for node_id in required_nodes
        if node_id.replace("\\", "/") not in collected_nodes
    )
    if missing_nodes:
        raise pytest.UsageError(
            "required impact-test nodes were not collected: "
            + ", ".join(missing_nodes)
        )

_IDENTITY_GROWTH_DATABASE_GUARD_ENV = (
    "IDENTITY_GROWTH_DATABASE_GUARD"
)
_IDENTITY_GROWTH_TEST_DATABASE_ENV = (
    "IDENTITY_GROWTH_TEST_DATABASE"
)

load_dotenv(override=False)
if os.environ.get(_IDENTITY_GROWTH_DATABASE_GUARD_ENV) == "1":
    identity_growth_database_name = os.environ.get(
        _IDENTITY_GROWTH_TEST_DATABASE_ENV,
        "",
    ).strip()
    if not identity_growth_database_name:
        raise RuntimeError(
            f"{_IDENTITY_GROWTH_TEST_DATABASE_ENV} must name the "
            "identity-growth test database"
        )
    if os.environ.get("MONGODB_DB_NAME") != identity_growth_database_name:
        raise RuntimeError(
            "MONGODB_DB_NAME must match the guarded identity-growth "
            "test database"
        )
    os.environ.pop("STAGE3_DATABASE_GUARD", None)
elif os.environ.get("MONGODB_DB_NAME") == STAGE3_TEST_DATABASE_NAME:
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
