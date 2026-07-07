"""Data models for the issue tracker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Issue:
    id: str
    title: str
    status: str = "open"
    assignee: str | None = None
