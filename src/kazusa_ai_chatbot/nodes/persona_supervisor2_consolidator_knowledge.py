"""Stage 4 consolidator knowledge-base distillation helpers."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.db import get_text_embedding
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import ConsolidatorState
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache
from kazusa_ai_chatbot.rag.depth_classifier import DEEP
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


_KNOWLEDGE_BASE_DISTILL_PROMPT = """\
你负责从本轮对话的信息检索结果中提取具有通用参考价值的知识条目，以便未来相似话题的对话可以复用。

# 提取准则
1. 仅保留**客观事实性知识**，不包含用户个人信息或角色主观看法。
2. 每条知识应能独立成立，脱离当前对话上下文仍有意义。
3. 避免提取已经是常识的信息。
4. 每条知识简洁陈述（60字以内）。
5. 若无值得提取的知识，返回空列表。

# 输入格式
{{
    "input_context_results": "本轮话题相关记忆检索结果",
    "external_rag_results": "本轮外部知识检索结果"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段:
{{
    "knowledge_entries": [
        "客观知识条目1",
        "客观知识条目2"
    ]
}}
"""
_knowledge_base_distill_llm = get_llm(temperature=0.0, top_p=1.0)


async def _update_knowledge_base(
    state: "ConsolidatorState",
) -> int:
    """Distil topic knowledge from this session's deep RAG results into the knowledge base.

    Only runs when the RAG metadata records a DEEP dispatch, and only when
    there is non-empty retrieval content to distil.

    Args:
        state: Consolidator state carrying ``metadata`` (with ``depth`` field
            from the RAG pass) and ``research_facts``.

    Returns:
        Number of knowledge entries stored (0 if nothing was written).
    """
    metadata = state.get("metadata") or {}
    if metadata.get("depth") != DEEP:
        return 0

    research_facts = state.get("research_facts") or {}
    input_context_results = research_facts.get("input_context_results") or ""
    external_results = research_facts.get("external_rag_results") or ""

    if not input_context_results and not external_results:
        return 0

    system_prompt = SystemMessage(_KNOWLEDGE_BASE_DISTILL_PROMPT)
    user_prompt = HumanMessage(content=json.dumps({
        "input_context_results": input_context_results,
        "external_rag_results": external_results,
    }, ensure_ascii=False))
    response = await _knowledge_base_distill_llm.ainvoke([system_prompt, user_prompt])
    result = parse_llm_json_output(response.content)
    entries: list[str] = result.get("knowledge_entries") or []

    if not entries:
        return 0

    rag_cache = await _get_rag_cache()
    stored = 0
    for entry in entries:
        if not entry:
            continue
        try:
            embedding = await get_text_embedding(entry)
            await rag_cache.store(
                embedding=embedding,
                results={"knowledge_base_results": entry},
                cache_type="knowledge_base",
                global_user_id="",
                metadata={"source": "knowledge_base_updater"},
            )
            stored += 1
        except Exception:
            logger.exception("_update_knowledge_base: failed to store entry")

    return stored
