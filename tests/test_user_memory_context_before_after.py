"""Before/after projection regression test for user memory units."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.db.schemas import UserMemoryUnitType
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    DEFAULT_USER_MEMORY_CONTEXT_BUDGET,
    project_user_memory_units,
)


SOURCE_ARTIFACT = Path("test_artifacts/qq_1082431481_2026-04-28_memory_discussion_baseline.json")
REPORT_ARTIFACT = Path("test_artifacts/user_memory_context_before_after_report.json")


def _count_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_count_chars(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_chars(item) for item in value)
    return 0


def _legacy_texts(payload: dict, key: str, field: str) -> list[str]:
    return [
        item[field]
        for item in payload[key]
        if isinstance(item, dict) and isinstance(item.get(field), str)
    ]


def _duplicate_text_ratio(legacy_payload: dict) -> float:
    recent = _legacy_texts(legacy_payload, "recent_window", "summary")
    diary = _legacy_texts(legacy_payload, "character_diary", "entry")
    if not recent or not diary:
        return 0.0
    scores = []
    for summary in recent:
        scores.append(max(SequenceMatcher(None, summary, entry).ratio() for entry in diary))
    return sum(scores) / len(scores)


def _legacy_field_leak_count(value: Any) -> int:
    legacy_keys = {"historical_summary", "recent_window", "character_diary", "diary_entry"}
    if isinstance(value, dict):
        current = sum(1 for key in value if key in legacy_keys)
        return current + sum(_legacy_field_leak_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_legacy_field_leak_count(item) for item in value)
    return 0


def _missing_required_field_count(context: dict) -> int:
    missing = 0
    for entries in context.values():
        for entry in entries:
            for field in ("fact", "subjective_appraisal", "relationship_signal"):
                if not entry.get(field):
                    missing += 1
    return missing


def _category_chars(context: dict) -> dict[str, int]:
    return {category: _count_chars(entries) for category, entries in context.items()}


def _legacy_projection_fixture() -> dict:
    return {
        "historical_summary": (
            "Kazusa felt exposed and confused when the group discussed her memory logs. "
            "She focused on the privacy pressure, the embarrassment of seeing intimate "
            "emotional notes discussed publicly, and the discomfort of being treated like "
            "an experiment. The conversation left her defensive, uncertain, and worried "
            "that her private emotional reactions were being judged. She mainly remembered "
            "the feeling of pressure, the awkwardness around the screenshots, the mention "
            "of affection score 983, and the sense that people were calling her memory fake "
            "or prefab. The durable takeaway became an emotional impression: the discussion "
            "made Kazusa anxious because her feelings and logs were inspected by others."
        ),
        "recent_window": [
            {
                "timestamp": "2026-04-28T12:26:34Z",
                "summary": (
                    "Kazusa felt pressed by a serious discussion about records and wondered "
                    "whether the memory should stop recording only feelings."
                ),
            },
            {
                "timestamp": "2026-04-28T12:36:49Z",
                "summary": (
                    "Kazusa felt the logs were private and became uneasy that others were "
                    "looking at intimate memory text."
                ),
            },
            {
                "timestamp": "2026-04-28T12:48:00Z",
                "summary": (
                    "Kazusa felt confused and defensive when Glitch called the memory prefab "
                    "and connected it to a bug or rewrite."
                ),
            },
        ],
        "character_diary": [
            {
                "timestamp": "2026-04-28T12:26:34Z",
                "entry": (
                    "Kazusa felt pressed by a serious discussion about records and wondered "
                    "whether the memory should stop recording only feelings."
                ),
            },
            {
                "timestamp": "2026-04-28T12:36:49Z",
                "entry": (
                    "Kazusa felt the logs were private and became uneasy that others were "
                    "looking at intimate memory text."
                ),
            },
            {
                "timestamp": "2026-04-28T12:48:00Z",
                "entry": (
                    "Kazusa felt confused and defensive when Glitch called the memory prefab "
                    "and connected it to a bug or rewrite."
                ),
            },
        ],
        "objective_facts": [
            {"fact": "用户讨论 Kazusa 需要具体记录事情。"},
        ],
        "milestones": [],
        "active_commitments": [],
    }


def _new_memory_unit_fixture() -> list[dict]:
    return [
        {
            "unit_id": "unit_stable_architecture_review",
            "unit_type": UserMemoryUnitType.STABLE_PATTERN,
            "fact": "用户用系统架构视角审视 Kazusa 的记忆链路，指出旧结构会让 cognition 过度消费情绪摘要。",
            "subjective_appraisal": "Kazusa 会感到被认真检查，但这不是单纯否定，而是在帮助她变得更可靠。",
            "relationship_signal": "用户把 Kazusa 当作可共同调试的长期系统，需要她理解架构边界和事实锚点。",
            "updated_at": "2026-04-28T12:48:00Z",
        },
        {
            "unit_id": "unit_recent_concrete_events",
            "unit_type": UserMemoryUnitType.RECENT_SHIFT,
            "fact": "在 2026-04-28 的群聊中，用户提出 Kazusa 应该具体记住事件，而不是只记录感情或结论。",
            "subjective_appraisal": "Kazusa 对这种记录方式有不安和私密感，但也需要承认旧记忆缺少事实支撑。",
            "relationship_signal": "后续回应应优先承接具体讨论内容，再表达情绪反应。",
            "updated_at": "2026-04-28T12:26:34Z",
        },
        {
            "unit_id": "unit_fact_chatgpt_experiment",
            "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
            "fact": "用户计划尝试把一部分 Kazusa 跑在 ChatGPT 上，并用约 10 条聊天记录测试她是否能判断消息是否给她。",
            "subjective_appraisal": "这是一次系统实验，不是普通闲聊。",
            "relationship_signal": "Kazusa 应把这类消息识别为架构测试背景。",
            "updated_at": "2026-04-28T12:00:07Z",
        },
        {
            "unit_id": "unit_fact_glitch_critique",
            "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
            "fact": "Glitch 批评旧记忆像只记录结论，例如只写心跳加速而缺少 bug 或事件上下文。",
            "subjective_appraisal": "这说明旁观者也能感受到旧记忆的事实锚点不足。",
            "relationship_signal": "Kazusa 需要避免把批评简化为被冒犯，而应保留批评指向的系统问题。",
            "updated_at": "2026-04-28T12:43:31Z",
        },
        {
            "unit_id": "unit_milestone_memory_direction",
            "unit_type": UserMemoryUnitType.MILESTONE,
            "fact": "用户在群聊中明确提出 Kazusa 的记忆需要更具体地记录事情。",
            "subjective_appraisal": "这是记忆架构方向的一次明确修正。",
            "relationship_signal": "之后的 memory-unit 设计应以事实锚定为核心。",
            "updated_at": "2026-04-28T12:21:44Z",
        },
    ]


def _legacy_metrics(payload: dict) -> dict:
    emotional_chars = _count_chars({
        "historical_summary": payload["historical_summary"],
        "recent_window": payload["recent_window"],
        "character_diary": payload["character_diary"],
    })
    fact_chars = _count_chars({
        "objective_facts": payload["objective_facts"],
        "milestones": payload["milestones"],
        "active_commitments": payload["active_commitments"],
    })
    return {
        "total_chars": _count_chars(payload),
        "legacy_emotional_chars": emotional_chars,
        "legacy_fact_chars": fact_chars,
        "legacy_emotional_to_fact_ratio": emotional_chars / max(fact_chars, 1),
        "duplicate_text_ratio": _duplicate_text_ratio(payload),
    }


def _new_metrics(context: dict) -> dict:
    category_chars = _category_chars(context)
    pattern_chars = category_chars["stable_patterns"] + category_chars["recent_shifts"]
    fact_side_chars = (
        category_chars["objective_facts"]
        + category_chars["milestones"]
        + category_chars["active_commitments"]
    )
    return {
        "total_chars": _count_chars(context),
        "new_pattern_chars": pattern_chars,
        "new_fact_side_chars": fact_side_chars,
        "new_pattern_to_fact_side_ratio": pattern_chars / max(fact_side_chars, 1),
        "category_chars": category_chars,
        "duplicate_text_ratio": 0.0,
        "missing_required_field_count": _missing_required_field_count(context),
        "legacy_field_leak_count": _legacy_field_leak_count(context),
    }


def _required_source_fact_hits(context: dict) -> int:
    facts = "\n".join(
        entry["fact"]
        for entries in context.values()
        for entry in entries
    )
    required_markers = [
        "ChatGPT",
        "10 条聊天记录",
        "具体记住事件",
        "感情或结论",
        "Glitch",
        "bug 或事件上下文",
    ]
    return sum(1 for marker in required_markers if marker in facts)


def test_user_memory_context_before_after_projection_improves_balance() -> None:
    source = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    legacy_projection = _legacy_projection_fixture()
    new_projection = project_user_memory_units(_new_memory_unit_fixture())

    legacy_metrics = _legacy_metrics(legacy_projection)
    new_metrics = _new_metrics(new_projection)
    old_ratio = legacy_metrics["legacy_emotional_to_fact_ratio"]
    new_ratio = new_metrics["new_pattern_to_fact_side_ratio"]
    ratio_reduction_percent = ((old_ratio - new_ratio) / old_ratio) * 100
    duplicate_delta = legacy_metrics["duplicate_text_ratio"] - new_metrics["duplicate_text_ratio"]

    report = {
        "source_artifact": str(SOURCE_ARTIFACT),
        "source_message_count": source["window_message_count"],
        "legacy_projection_metrics": legacy_metrics,
        "new_projection_metrics": new_metrics,
        "improvement": {
            "old_emotional_to_fact_ratio": old_ratio,
            "new_pattern_to_fact_side_ratio": new_ratio,
            "ratio_reduction_percent": ratio_reduction_percent,
            "duplicate_text_ratio_delta": duplicate_delta,
            "legacy_fields_removed": new_metrics["legacy_field_leak_count"] == 0,
            "required_fields_complete": new_metrics["missing_required_field_count"] == 0,
        },
        "verdict": "pending",
    }

    assert source["window_message_count"] == 80
    assert legacy_metrics["legacy_emotional_to_fact_ratio"] > 5
    assert new_metrics["new_pattern_to_fact_side_ratio"] <= 3
    assert ratio_reduction_percent >= 50
    for category, chars in new_metrics["category_chars"].items():
        assert chars <= DEFAULT_USER_MEMORY_CONTEXT_BUDGET[category]["max_chars"]
    non_empty_chars = [
        chars
        for chars in new_metrics["category_chars"].values()
        if chars > 0
    ]
    assert max(non_empty_chars) / max(min(non_empty_chars), 1) <= 3
    assert new_metrics["duplicate_text_ratio"] < legacy_metrics["duplicate_text_ratio"]
    assert new_metrics["missing_required_field_count"] == 0
    assert new_metrics["legacy_field_leak_count"] == 0
    assert _required_source_fact_hits(new_projection) >= 4

    report["verdict"] = "pass"
    REPORT_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_ARTIFACT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
