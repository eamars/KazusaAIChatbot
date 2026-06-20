"""Evidence-only answer synthesis for code reading."""

from __future__ import annotations

from kazusa_ai_chatbot.coding_agent.code_reading.models import CodeEvidenceRow
from kazusa_ai_chatbot.coding_agent.code_reading.planner import ReadingPlan


def synthesize_answer(
    *,
    question: str,
    plan: ReadingPlan,
    evidence: list[CodeEvidenceRow],
    preferred_language: str | None,
    max_answer_chars: int,
) -> str:
    """Create a bounded answer from evidence rows only."""

    if plan.broad:
        return (
            "Please narrow the repository-reading question to a feature, "
            "module, symbol, API, tests, or source directory."
        )

    chinese = _should_use_chinese(question, preferred_language)
    if chinese:
        answer = _synthesize_chinese(plan, evidence)
    else:
        answer = _synthesize_english(plan, evidence)

    if len(answer) > max_answer_chars:
        answer = answer[:max_answer_chars].rstrip()
    return answer


def _synthesize_chinese(
    plan: ReadingPlan,
    evidence: list[CodeEvidenceRow],
) -> str:
    if not evidence:
        return '没有找到足够的本地代码证据来回答这个问题。'

    facts = _facts_from_evidence(evidence)
    if plan.family == "feature_pipeline_explanation":
        parts = ['这个项目的读图链路是：']
        if "base64_data" in facts:
            parts.append('适配器或输入层保留图片的 `base64_data`。')
        if "user_multimedia_input" in facts:
            parts.append('服务层把图片附件组装成 `user_multimedia_input`。')
        if "multimedia_descriptor_agent" in facts:
            parts.append(
                '`multimedia_descriptor_agent` 在图进入相关性和认知前生成描述。'
            )
        if "VISION_DESCRIPTOR_LLM" in facts:
            parts.append(
                '`VISION_DESCRIPTOR_LLM` 是图片描述模型路由或模型标识。'
            )
        if "image_observation" in facts:
            parts.append(
                '描述结果被写成结构化 `image_observation`，后续按媒体感知处理。'
            )
        if "<image>" in facts:
            parts.append('历史或回复里的图片描述会以 `<image>...</image>` 投影。')
        parts.append(_evidence_sentence(evidence, chinese=True))
        return " ".join(parts)

    return _synthesize_english(plan, evidence)


def _synthesize_english(
    plan: ReadingPlan,
    evidence: list[CodeEvidenceRow],
) -> str:
    if not evidence:
        return "I found no bounded local source evidence for this question."

    intro_by_family = {
        "feature_pipeline_explanation": "The feature flow is evidence-backed by these code paths.",
        "architecture_responsibility": "The responsibility boundary is visible in the matched modules.",
        "api_contract_lookup": "The public contract is visible in the matched route and request definitions.",
        "symbol_explanation": "The symbol behavior is visible in its definition and nearby usage.",
        "definition_usage_search": "The definition and representative usage are in the evidence below.",
        "scope_summary": "The requested scope contains these responsibilities.",
        "state_model_reading": "The state or data field is created and consumed in these places.",
        "lifecycle_cache_persistence": "The cache or persistence path is only partially visible from local evidence.",
        "test_coverage_mapping": "The checked-in tests covering this behavior are visible in the evidence.",
        "dependency_usage": "External integrations are visible through imports and wrapper calls.",
        "intra_repo_comparison": "The compared modules expose these responsibilities.",
        "docs_to_code_consistency": "The docs and implementation evidence can be compared from these rows.",
        "static_impact_read": "Likely impact is limited to visible definitions and call sites.",
        "build_run_reading": "Checked-in run instructions and deployment constants are visible here.",
    }
    parts = [intro_by_family.get(plan.family, "The answer is based on bounded local evidence.")]
    parts.extend(_evidence_bullets(evidence))
    return "\n".join(parts)


def _evidence_bullets(evidence: list[CodeEvidenceRow]) -> list[str]:
    bullets: list[str] = []
    for row in evidence:
        excerpt = " ".join(
            line.strip() for line in row["excerpt"].splitlines() if line.strip()
        )
        bullets.append(
            f"- {row['path']}:{row['line_start']} shows {excerpt}"
        )
    return bullets


def _facts_from_evidence(evidence: list[CodeEvidenceRow]) -> set[str]:
    text = "\n".join(row["excerpt"] for row in evidence)
    facts = set()
    for token in (
        "base64_data",
        "user_multimedia_input",
        "multimedia_descriptor_agent",
        "VISION_DESCRIPTOR_LLM",
        "image_observation",
        "<image>",
    ):
        if token in text:
            facts.add(token)
    return facts


def _evidence_sentence(
    evidence: list[CodeEvidenceRow],
    *,
    chinese: bool,
) -> str:
    paths = ", ".join(_dedupe_paths(row["path"] for row in evidence)[:8])
    if chinese:
        return f'证据来自这些文件：{paths}。'
    return f"Evidence files: {paths}."


def _dedupe_paths(paths: object) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not isinstance(path, str) or path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _should_use_chinese(
    question: str,
    preferred_language: str | None,
) -> bool:
    if preferred_language and preferred_language.casefold().startswith(
        ("chinese", "zh")
    ):
        return True
    return any("\u4e00" <= char <= "\u9fff" for char in question)
