"""Project maintenance scripts exposed from both repo and source layouts."""

from __future__ import annotations

from pathlib import Path

_SRC_SCRIPTS_PATH = Path(__file__).resolve().parents[1] / "src" / "scripts"
if _SRC_SCRIPTS_PATH.is_dir():
    _src_scripts_path = str(_SRC_SCRIPTS_PATH)
    if _src_scripts_path not in __path__:
        __path__.append(_src_scripts_path)
