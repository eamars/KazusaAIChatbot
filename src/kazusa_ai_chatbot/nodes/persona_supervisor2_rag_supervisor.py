"""Phase 5 — Bounded RAG Supervisor: evaluator with capped repair pass.

The evaluator inspects retrieval results after the first retrieval pass and
decides whether a single repair pass is needed.  Repair is scoped: it only
re-runs Tier 1-3 for newly revealed entities, not a full re-plan.

Control limits (from the design doc):
  - max evaluator-triggered repair pass: 1
  - max total new Stage-1 LLM rounds after decontextualizer: 2
  - no repeated ledger key in one request
  - repair pass scope: fresh Tier 1-3 cycle for newly revealed target only
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_schema import RAGState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

MAX_REPAIR_PASSES = 1


def _normalize_surface_form(value: str) -> str:
    """Normalize an entity surface form for repair-pass deduplication."""
    return str(value or "").strip().lower()

_RAG_EVALUATOR_PROMPT = """\
你是一个检索结果评估器。你的任务是判断当前的检索结果是否足以回答用户的查询，或者是否需要进行一次补充检索。

# 评估标准
1. **覆盖率**: 检索计划中列出的所有活跃源是否都有返回结果？
2. **相关性**: 返回的结果是否与解析后的任务相关？
3. **实体完整性**: 所有被识别的实体是否都有对应的检索结果？
4. **新实体发现**: 检索结果中是否发现了检索计划未预见到的新实体？

# 修复范围
如果需要修复，只能请求对**新发现的实体**进行补充检索，不能重新规划整个检索。

# 输入格式
{{
    "resolved_task": "解析后的完整任务",
    "retrieval_plan": {{...}},
    "resolved_entities": [...],
    "retrieval_ledger": {{...}},
    "result_summary": {{
        "input_context_results": "是否有结果",
        "channel_recent_entity_results": "是否有结果",
        "third_party_profile_results": "是否有结果",
        "entity_knowledge_results": "是否有结果",
        "external_rag_results": "是否有结果"
    }},
    "repair_pass": 0
}}

# 输出格式
请务必返回合法的 JSON 字符串：
{{
    "verdict": "sufficient | needs_repair",
    "reasoning": "评估理由",
    "coverage_score": 0.0到1.0,
    "missing_entities": ["需要补充检索的新实体名称"],
    "repair_sources": ["需要补充的检索源列表"]
}}

# 约束
- 如果 repair_pass >= 1，必须返回 "sufficient"，不允许再次修复
- 只有在关键信息缺失且补充检索有可能改善结果时，才返回 "needs_repair"
- 对于简单查询（日常对话、情感互动），应直接返回 "sufficient"
"""

_evaluator_llm = get_llm(temperature=0.0, top_p=1.0)


async def rag_supervisor_evaluator(state: RAGState) -> dict:
    """Evaluate retrieval results and decide whether a repair pass is needed.

    Returns a dict with:
      - ``evaluation``: the full evaluation result
      - ``needs_repair``: bool
      - ``repair_entities``: list of new entity surface forms to fetch
      - ``repair_sources``: list of source types to re-run
    """
    repair_pass = state.get("rag_metadata", {}).get("repair_pass", 0)

    # Hard cap: no repair after max passes
    if repair_pass >= MAX_REPAIR_PASSES:
        return {
            "evaluation": {
                "verdict": "sufficient",
                "reasoning": "Repair pass limit reached",
                "coverage_score": 1.0,
                "missing_entities": [],
                "repair_sources": [],
            },
            "needs_repair": False,
            "repair_entities": [],
            "repair_sources": [],
        }

    continuation = state.get("continuation_context") or {}
    retrieval_plan = state.get("retrieval_plan") or {}
    resolved_entities = state.get("resolved_entities") or []
    ledger = state.get("retrieval_ledger") or {}

    # Build result summary for the evaluator
    result_summary = {
        "input_context_results": bool(state.get("input_context_results")),
        "channel_recent_entity_results": bool(state.get("channel_recent_entity_results")),
        "third_party_profile_results": bool(state.get("third_party_profile_results")),
        "entity_knowledge_results": bool(state.get("entity_knowledge_results")),
        "external_rag_results": bool(state.get("external_rag_results")),
    }

    # Quick exit: if retrieval mode is NONE or CURRENT_USER_STABLE, no evaluation needed
    mode = retrieval_plan.get("retrieval_mode", "NONE")
    if mode in ("NONE", "CURRENT_USER_STABLE"):
        return {
            "evaluation": {
                "verdict": "sufficient",
                "reasoning": f"Retrieval mode is {mode}, no evaluation needed",
                "coverage_score": 1.0,
                "missing_entities": [],
                "repair_sources": [],
            },
            "needs_repair": False,
            "repair_entities": [],
            "repair_sources": [],
        }

    system_prompt = SystemMessage(content=_RAG_EVALUATOR_PROMPT)
    user_input = {
        "resolved_task": continuation.get("resolved_task", ""),
        "retrieval_plan": {
            "retrieval_mode": retrieval_plan.get("retrieval_mode", ""),
            "active_sources": retrieval_plan.get("active_sources", []),
            "entities": retrieval_plan.get("entities", []),
        },
        "resolved_entities": [
            {
                "surface_form": e.get("surface_form", ""),
                "resolved_global_user_id": bool(e.get("resolved_global_user_id")),
                "resolution_confidence": e.get("resolution_confidence", 0),
            }
            for e in resolved_entities
        ],
        "retrieval_ledger": {k: v for k, v in ledger.items()},
        "result_summary": result_summary,
        "repair_pass": repair_pass,
    }

    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    try:
        response = await _evaluator_llm.ainvoke([system_prompt, human_message])
        result = parse_llm_json_output(response.content)
    except Exception:
        logger.warning("RAG evaluator LLM call failed, defaulting to sufficient", exc_info=True)
        result = {
            "verdict": "sufficient",
            "reasoning": "Evaluator LLM failed, defaulting to sufficient",
            "coverage_score": 0.5,
            "missing_entities": [],
            "repair_sources": [],
        }

    verdict = str(result.get("verdict", "sufficient")).lower()
    needs_repair = verdict == "needs_repair"
    missing_entities = result.get("missing_entities", [])
    repair_sources = result.get("repair_sources", [])
    planned_entities = {
        _normalize_surface_form(entity.get("surface_form", ""))
        for entity in retrieval_plan.get("entities", [])
    }

    # Enforce the Phase-5 contract: repair may only target newly revealed entities.
    filtered_entities = [
        entity for entity in missing_entities
        if _normalize_surface_form(entity) not in planned_entities
    ]
    if needs_repair and not filtered_entities:
        needs_repair = False
        result["reasoning"] = (
            result.get("reasoning", "")
            + " (overridden: repair requires newly revealed entities)"
        )

    # Enforce: don't re-fetch ledger keys
    filtered_entities = [
        e for e in filtered_entities
        if not any(e.lower() in k.lower() for k in ledger)
    ]
    if needs_repair and not filtered_entities:
        needs_repair = False
        result["reasoning"] = (result.get("reasoning", "") + " (overridden: all missing entities already in ledger)")

    logger.debug(
        "RAG evaluator: verdict=%s coverage=%.2f needs_repair=%s missing=%s repair_sources=%s",
        verdict,
        float(result.get("coverage_score", 0)),
        needs_repair,
        filtered_entities,
        repair_sources,
    )

    return {
        "evaluation": result,
        "needs_repair": needs_repair,
        "repair_entities": filtered_entities,
        "repair_sources": repair_sources,
    }
