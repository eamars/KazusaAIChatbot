"""Validate and render the inspected real-history E2E evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CASE_COUNT = 20


def _load_case_files(profile_root: Path) -> dict[str, dict[str, Any]]:
    """Load all successful case artifacts for one profile."""

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(profile_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"案例证据不是对象: {path}")
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"案例证据缺少 case_id: {path}")
        if payload.get("schema_version") != (
            "real_history_personality_e2e_case.v2"
        ):
            raise ValueError(f"案例证据 schema 不是 v2: {path}")
        if payload.get("technical_status") != "passed":
            raise ValueError(f"案例技术状态不是通过: {path}")
        if payload.get("source_kind") != "real_conversation_history_export":
            raise ValueError(f"案例来源不是 conversation_history: {path}")
        fixture_validity = payload.get("fixture_validity")
        if (
            not isinstance(fixture_validity, dict)
            or fixture_validity.get("passed") is not True
        ):
            raise ValueError(f"案例夹具校验不是通过: {path}")
        semantic_validity = payload.get("semantic_validity")
        if (
            not isinstance(semantic_validity, dict)
            or semantic_validity.get("passed") is not True
        ):
            raise ValueError(f"案例语义校验不是通过: {path}")
        if not isinstance(payload.get("source_input"), dict):
            raise ValueError(f"案例缺少 source_input: {path}")
        if not isinstance(payload.get("effective_input"), dict):
            raise ValueError(f"案例缺少 effective_input: {path}")
        if not isinstance(payload.get("identity_mapping"), list):
            raise ValueError(f"案例缺少 identity_mapping: {path}")
        if not isinstance(payload.get("excluded_rows"), list):
            raise ValueError(f"案例缺少 excluded_rows: {path}")
        if not isinstance(payload.get("effective_context"), list):
            raise ValueError(f"案例缺少 effective_context: {path}")
        if not isinstance(payload.get("decontextualized_input"), str):
            raise ValueError(f"案例缺少 decontextualized_input: {path}")
        if not isinstance(payload.get("private_monologue"), str):
            raise ValueError(f"案例缺少 private_monologue: {path}")
        if not isinstance(payload.get("visible_dialog"), list):
            raise ValueError(f"案例缺少 visible_dialog: {path}")
        if case_id in artifacts:
            raise ValueError(f"案例重复: {case_id}")
        artifacts[case_id] = payload
    if len(artifacts) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"{profile_root} 需要 {EXPECTED_CASE_COUNT} 个案例，"
            f"实际为 {len(artifacts)} 个"
        )
    return artifacts


def _render_text(value: object, empty_label: str) -> str:
    """Render one raw monologue or visible message list without translation."""

    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(str(item) for item in value if str(item).strip())
    else:
        text = ""
    return text if text.strip() else empty_label


def build_report(root: Path, output_path: Path) -> None:
    """Pair the two profile runs and write only input, monologue, and dialog."""

    asuna_cases = _load_case_files(root / "asuna")
    kazusa_cases = _load_case_files(root / "kazusa")
    if set(asuna_cases) != set(kazusa_cases):
        raise ValueError("两种人格的案例集合不一致")
    first_case = next(iter(asuna_cases.values()))
    source_indexes = {
        case_id: payload.get("source_index")
        for case_id, payload in asuna_cases.items()
    }
    if source_indexes != {
        case_id: payload.get("source_index")
        for case_id, payload in kazusa_cases.items()
    }:
        raise ValueError("两种人格的来源行不一致")
    source_inputs = {
        case_id: payload["source_input"].get("body_text")
        for case_id, payload in asuna_cases.items()
    }
    if source_inputs != {
        case_id: payload["source_input"].get("body_text")
        for case_id, payload in kazusa_cases.items()
    }:
        raise ValueError("两种人格的源用户输入不一致")
    asuna_name = str(
        asuna_cases[next(iter(asuna_cases))].get(
            "character_semantic_name",
            "一之濑明日奈",
        )
    )
    kazusa_name = str(
        kazusa_cases[next(iter(kazusa_cases))].get(
            "character_semantic_name",
            "杏山千纱",
        )
    )

    lines = [
        "# 实际对话历史端到端人格对比报告",
        "",
        "本报告只列出历史源用户输入、两侧实际用户输入、私人独白和可见对话。",
        "",
    ]
    for case_id in sorted(asuna_cases):
        asuna = asuna_cases[case_id]
        kazusa = kazusa_cases[case_id]
        source_input = asuna["source_input"].get("body_text", "")
        asuna_input = asuna["effective_input"].get("body_text", "")
        kazusa_input = kazusa["effective_input"].get("body_text", "")
        lines.extend([
            f"## 案例 {case_id}",
            "",
            "### 历史源用户输入",
            "",
            str(source_input),
            "",
            f"### {asuna_name}",
            "",
            "#### 本轮用户输入",
            "",
            str(asuna_input),
            "",
            "#### 私人独白",
            "",
            _render_text(asuna.get("private_monologue"), "（未捕获私人独白）"),
            "",
            "#### 可见对话",
            "",
            _render_text(asuna.get("visible_dialog"), "（无可见输出）"),
            "",
            f"### {kazusa_name}",
            "",
            "#### 本轮用户输入",
            "",
            str(kazusa_input),
            "",
            "#### 私人独白",
            "",
            _render_text(
                kazusa.get("private_monologue"),
                "（未捕获私人独白）",
            ),
            "",
            "#### 可见对话",
            "",
            _render_text(kazusa.get("visible_dialog"), "（无可见输出）"),
            "",
        ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Parse report paths and write the consolidated comparison artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.root.resolve(), args.output.resolve())
    print(f"报告已生成: {args.output.resolve()}")


if __name__ == "__main__":
    main()
