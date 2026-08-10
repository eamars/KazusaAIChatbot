"""Audit live LLM producer call sites in both frozen source trees."""

from __future__ import annotations

import argparse
import ast
from fnmatch import fnmatchcase
import json
from pathlib import Path
import sys
from typing import Any


class _CallSiteVisitor(ast.NodeVisitor):
    """Collect semantic LLM invocation call sites with stable local ordinals."""

    def __init__(self) -> None:
        self._scope: list[str] = []
        self._ordinal = 0
        self.rows: list[dict[str, Any]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function with its qualified scope."""

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a synchronous function with its qualified scope."""

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Include class scope in nested function names."""

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Record one direct LLM method or project LLM wrapper call."""

        call_kind = _call_kind(node)
        if call_kind:
            self._ordinal += 1
            qualified_function = ".".join(self._scope) or "<module>"
            self.rows.append({
                "line": node.lineno,
                "column": node.col_offset,
                "qualified_function": qualified_function,
                "call_kind": call_kind,
                "call_ordinal": self._ordinal,
            })
        self.generic_visit(node)


def _call_kind(node: ast.Call) -> str:
    """Return the producer call kind or an empty string."""

    function = node.func
    if isinstance(function, ast.Attribute):
        if function.attr in {"ainvoke", "invoke"}:
            return function.attr
        if (
            function.attr.startswith("run_")
            and function.attr.endswith("_llm")
        ):
            return function.attr
    if isinstance(function, ast.Name):
        if function.id.startswith("run_") and function.id.endswith("_llm"):
            return function.id
    return ""


def _load_matrix(path: Path) -> dict[str, Any]:
    """Load the producer matrix fixture."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("producer matrix root must be an object")
    return value


def _relative_source_path(source_root: Path, path: Path) -> str:
    """Return a stable repository-relative source path."""

    relative_path = path.relative_to(source_root).as_posix()
    return f"src/{relative_path}"


def _discover_call_sites(target_root: Path) -> list[dict[str, Any]]:
    """Parse every target Python source file and return call-site rows."""

    source_root = target_root / "src"
    package_root = source_root / "kazusa_ai_chatbot"
    if not package_root.is_dir():
        raise ValueError(f"target source package is missing: {package_root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*.py")):
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(path))
        visitor = _CallSiteVisitor()
        visitor.visit(tree)
        relative_path = _relative_source_path(source_root, path)
        for row in visitor.rows:
            call_site = dict(row)
            call_site["relative_path"] = relative_path
            call_site["call_site_id"] = (
                f"{relative_path}:{row['qualified_function']}:"
                f"{row['call_ordinal']}"
            )
            rows.append(call_site)
    return rows


def _matching_rules(
    relative_path: str,
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the matrix rules matching one source path."""

    return [
        rule
        for rule in rules
        if fnmatchcase(relative_path, str(rule.get("pattern", "")))
    ]


def audit(
    *,
    revision: str,
    target_root: Path,
    matrix_path: Path,
    output_path: Path | None,
    quiet: bool,
) -> int:
    """Audit one frozen target tree and write an evidence report."""

    matrix = _load_matrix(matrix_path)
    rules = matrix.get("call_site_rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("producer matrix has no call_site_rules")
    call_sites = _discover_call_sites(target_root.resolve())
    unmatched: list[dict[str, Any]] = []
    duplicate_matches: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    rule_usage: dict[str, int] = {
        str(rule["pattern"]): 0
        for rule in rules
    }
    for call_site in call_sites:
        matches = _matching_rules(call_site["relative_path"], rules)
        if not matches:
            unmatched.append(call_site)
            continue
        if len(matches) != 1:
            duplicate_matches.append({
                "call_site": call_site,
                "patterns": [str(rule.get("pattern")) for rule in matches],
            })
            continue
        rule = matches[0]
        rule_usage[str(rule["pattern"])] += 1
        matched_rows.append({
            **call_site,
            "owner": rule["owner"],
            "route": rule["route"],
            "parser": rule["parser"],
            "attempt_cap": rule["attempt_cap"],
            "fault_selector": rule["fault_selector"],
        })
    evidence = {
        "schema_version": "cognition_llm_producer_inventory.v1",
        "revision": revision,
        "target_root": str(target_root.resolve()),
        "matrix_path": str(matrix_path.resolve()),
        "call_site_count": len(call_sites),
        "matched_call_site_count": len(matched_rows),
        "unmatched_call_sites": unmatched,
        "duplicate_rule_matches": duplicate_matches,
        "unused_rules": [
            pattern
            for pattern, count in rule_usage.items()
            if count == 0
        ],
        "rule_usage": rule_usage,
        "call_sites": matched_rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if quiet:
        print(json.dumps({
            "revision": revision,
            "call_site_count": len(call_sites),
            "matched_call_site_count": len(matched_rows),
            "unmatched_count": len(unmatched),
            "duplicate_rule_match_count": len(duplicate_matches),
        }, ensure_ascii=False))
    else:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    if unmatched or duplicate_matches:
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the producer inventory command-line parser."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--revision", required=True)
    audit_parser.add_argument("--target-root", required=True, type=Path)
    audit_parser.add_argument("--matrix", required=True, type=Path)
    audit_parser.add_argument("--output", type=Path)
    audit_parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested producer inventory command."""

    args = _build_parser().parse_args(argv)
    if args.command == "audit":
        return audit(
            revision=args.revision,
            target_root=args.target_root,
            matrix_path=args.matrix,
            output_path=args.output,
            quiet=args.quiet,
        )
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
