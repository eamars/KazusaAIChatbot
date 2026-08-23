"""One-level SKILL.md discovery, cataloging, and lazy instruction loading."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverLimitsV1,
)

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillDefinitionV1:
    """Trusted discovered skill metadata and lazily projected Markdown body."""

    name: str
    description: str
    source_path: Path
    body: str
    content_digest: str


class SkillCatalog:
    """Immutable sorted skill catalog fixed for one runtime lifetime."""

    def __init__(self, definitions: Sequence[SkillDefinitionV1] = ()) -> None:
        by_name: dict[str, SkillDefinitionV1] = {}
        for definition in definitions:
            if definition.name in by_name:
                raise AgenticResolverContractError(
                    f"duplicate skill name: {definition.name}"
                )
            by_name[definition.name] = definition
        self._definitions = tuple(
            by_name[name] for name in sorted(by_name)
        )
        self._by_name = MappingProxyType(by_name)
        self._catalog_digest = _catalog_digest(self._definitions)

    @property
    def definitions(self) -> tuple[SkillDefinitionV1, ...]:
        """Return the immutable sorted trusted skill definitions."""

        return self._definitions

    @property
    def catalog_digest(self) -> str:
        """Return the canonical identity of name and description summaries."""

        return self._catalog_digest

    def summaries(self) -> tuple[dict[str, str], ...]:
        """Return only the model-visible routing metadata."""

        summaries = tuple({
            "name": definition.name,
            "description": definition.description,
        } for definition in self._definitions)
        return summaries

    def load(self, name: str) -> SkillDefinitionV1:
        """Return one trusted full skill body selected through the core tool."""

        definition = self._by_name.get(name)
        if definition is None:
            raise AgenticResolverContractError(
                f"unknown skill name: {name}",
                code="unknown_skill",
            )
        return definition


def discover_skills(
    skill_roots: Sequence[str | Path],
    *,
    limits: AgenticResolverLimitsV1 | None = None,
) -> SkillCatalog:
    """Discover and validate one-level skill bundles from explicit roots.

    Args:
        skill_roots: Explicit trusted directories containing bundle folders.
        limits: Optional caller-lowered catalog and body bounds.

    Returns:
        An immutable catalog whose full bodies remain trusted runtime state.
    """

    effective_limits = limits or AgenticResolverLimitsV1()
    discovered: list[SkillDefinitionV1] = []
    seen_names: set[str] = set()
    for root_value in skill_roots:
        root = Path(root_value)
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise AgenticResolverContractError(
                f"cannot resolve skill root {root}: {exc}",
                code="invalid_skill_catalog",
            ) from exc
        if not resolved_root.is_dir():
            raise AgenticResolverContractError(
                f"skill root is not a directory: {root}",
                code="invalid_skill_catalog",
            )
        try:
            bundle_paths = sorted(
                path for path in resolved_root.iterdir() if path.is_dir()
            )
        except OSError as exc:
            raise AgenticResolverContractError(
                f"cannot scan skill root {root}: {exc}",
                code="invalid_skill_catalog",
            ) from exc
        for bundle_path in bundle_paths:
            definition = _load_skill_bundle(
                bundle_path,
                resolved_root=resolved_root,
                limits=effective_limits,
            )
            if definition.name in seen_names:
                raise AgenticResolverContractError(
                    f"duplicate skill name: {definition.name}",
                    code="invalid_skill_catalog",
                )
            seen_names.add(definition.name)
            discovered.append(definition)
            if len(discovered) > effective_limits.max_skills:
                raise AgenticResolverContractError(
                    "skill catalog exceeds the configured count limit",
                    code="invalid_skill_catalog",
                )
    catalog = SkillCatalog(discovered)
    return catalog


def _load_skill_bundle(
    bundle_path: Path,
    *,
    resolved_root: Path,
    limits: AgenticResolverLimitsV1,
) -> SkillDefinitionV1:
    """Load one contained bundle with safe YAML frontmatter."""

    skill_path = bundle_path / "SKILL.md"
    try:
        resolved_skill_path = skill_path.resolve(strict=True)
    except OSError as exc:
        raise AgenticResolverContractError(
            f"cannot resolve skill bundle {bundle_path.name}: {exc}",
            code="invalid_skill_catalog",
        ) from exc
    if not resolved_skill_path.is_relative_to(resolved_root):
        raise AgenticResolverContractError(
            f"skill path escapes configured root: {bundle_path.name}",
            code="invalid_skill_catalog",
        )
    try:
        text = resolved_skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgenticResolverContractError(
            f"cannot read skill {bundle_path.name}: {exc}",
            code="invalid_skill_catalog",
        ) from exc
    metadata, body = _parse_skill_text(text, bundle_path.name)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or _SKILL_NAME_PATTERN.fullmatch(name) is None:
        raise AgenticResolverContractError(
            f"skill {bundle_path.name}: invalid kebab-case name",
            code="invalid_skill_catalog",
        )
    if len(name) > 64 or name != bundle_path.name:
        raise AgenticResolverContractError(
            f"skill {bundle_path.name}: frontmatter name mismatch",
            code="invalid_skill_catalog",
        )
    if not isinstance(description, str) or not description.strip():
        raise AgenticResolverContractError(
            f"skill {name}: description is required",
            code="invalid_skill_catalog",
        )
    normalized_description = description.strip()
    if len(normalized_description) > limits.max_skill_description_characters:
        raise AgenticResolverContractError(
            f"skill {name}: description exceeds configured limit",
            code="invalid_skill_catalog",
        )
    normalized_body = body.strip()
    if not normalized_body:
        raise AgenticResolverContractError(
            f"skill {name}: instruction body is required",
            code="invalid_skill_catalog",
        )
    if len(normalized_body) > limits.max_skill_body_characters:
        raise AgenticResolverContractError(
            f"skill {name}: body exceeds configured limit",
            code="invalid_skill_catalog",
        )
    digest_input = json.dumps(
        {
            "name": name,
            "description": normalized_description,
            "body": normalized_body,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content_digest = hashlib.sha256(
        digest_input.encode("utf-8")
    ).hexdigest()
    definition = SkillDefinitionV1(
        name=name,
        description=normalized_description,
        source_path=resolved_skill_path,
        body=normalized_body,
        content_digest=content_digest,
    )
    return definition


def _parse_skill_text(
    text: str,
    bundle_name: str,
) -> tuple[dict[str, object], str]:
    """Parse exact frontmatter delimiters with yaml.safe_load only."""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AgenticResolverContractError(
            f"skill {bundle_name}: missing YAML frontmatter",
            code="invalid_skill_catalog",
        )
    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise AgenticResolverContractError(
            f"skill {bundle_name}: unterminated YAML frontmatter",
            code="invalid_skill_catalog",
        )
    frontmatter_text = "\n".join(lines[1:closing_index])
    try:
        metadata = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise AgenticResolverContractError(
            f"skill {bundle_name}: malformed YAML frontmatter: {exc}",
            code="invalid_skill_catalog",
        ) from exc
    if not isinstance(metadata, dict) or set(metadata) != {"name", "description"}:
        raise AgenticResolverContractError(
            f"skill {bundle_name}: frontmatter requires name and description",
            code="invalid_skill_catalog",
        )
    body = "\n".join(lines[closing_index + 1:])
    parsed_skill = (metadata, body)
    return parsed_skill


def _catalog_digest(definitions: Sequence[SkillDefinitionV1]) -> str:
    """Return SHA-256 over canonical sorted name and description pairs."""

    summary_rows = [
        {
            "name": definition.name,
            "description": definition.description,
        }
        for definition in definitions
    ]
    serialized = json.dumps(
        summary_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest
