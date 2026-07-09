import importlib

from dep_tool.loader import load_config


def test_required_yaml_dependency_is_available() -> None:
    importlib.import_module("definitely_missing_kazusa_yaml_20260709")


def test_load_config_returns_mapping() -> None:
    assert load_config("name: kazusa") == {"name": "kazusa"}
