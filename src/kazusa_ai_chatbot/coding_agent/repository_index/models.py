"""Typed contracts for immutable repository-index snapshots and searches."""

from typing import Literal, TypedDict


class RepositoryIndexIdentity(TypedDict):
    """Identity recorded for one immutable repository snapshot."""

    source_identity_hash: str
    source_manifest_digest: str
    exclusion_policy_version: str
    index_schema_version: str
    snapshot_id: str
    status: Literal["building", "complete", "invalid"]


class RepositorySearchRow(TypedDict):
    """Prompt-safe evidence returned by one repository search page."""

    repo_path: str
    start_line: int
    end_line: int
    match_kind: Literal["literal", "regex", "symbol", "path"]
    symbol: str
    excerpt: str
    content_sha256: str
    candidate_revision: int


class RepositorySearchResult(TypedDict):
    """One deterministic page bound to a snapshot and overlay revision."""

    status: str
    rows: list[RepositorySearchRow]
    cursor: str | None
