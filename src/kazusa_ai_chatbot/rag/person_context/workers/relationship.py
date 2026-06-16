
"""RAG helper agent: read character relationship rankings."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.db import list_users_by_affinity
from kazusa_ai_chatbot.rag.cache2_policy import (
    RELATIONSHIP_CACHE_NAME,
    build_relationship_cache_key,
    build_relationship_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.utils import build_affinity_block, parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
_MAX_RELATIONSHIP_LIMIT = 5
_RELATIONSHIP_MODES = {"one", "n", "existence"}
_RELATIONSHIP_RANK_ORDERS = {"top", "bottom"}

_NEGATIVE_LABELS = {
    "Contemptuous",
    "Scornful",
    "Hostile",
    "Antagonistic",
    "Aloof",
    "Reserved",
    "Formal",
    "Cold",
    "Detached",
}
_POSITIVE_LABELS = {
    "Receptive",
    "Approachable",
    "Friendly",
    "Warm",
    "Caring",
    "Affectionate",
    "Devoted",
    "Protective",
    "Fiercely Loyal",
    "Unwavering",
}

_EXTRACTOR_PROMPT = '''\
你是 `relationship_agent` 的参数抽取器。

# 能力边界
本代理按活跃角色的内部关系分数对已建档用户排序。适用于：
- 角色最喜欢谁、最偏爱谁、最亲近谁
- 是否存在被喜欢的人
- 角色最讨厌或最不喜欢谁
- top-N 关系排名

# 参数含义
- mode:
  - "one": 一个最强匹配候选
  - "n": 按请求数量返回排名列表
  - "existence": 判断是否存在匹配候选
- rank_order:
  - "top": 关系分数最高
  - "bottom": 关系分数最低
- limit:
  - 保留用户明确请求的数量。
  - mode="one" 时使用 1。
  - mode="existence" 默认使用 3，除非任务明确要求其他数量。

# 生成步骤
1. 读取 `task`，判断它要一个用户、排名列表还是存在性判断。
2. 选择 `rank_order`：喜欢/亲近/偏爱用 top；讨厌/不喜欢用 bottom。
3. 将明确数量写入 `limit`；否则使用 mode 默认值。
4. 忽略需要聊天证据或超出关系排名的人格解释任务。

# 输入格式
{
  "task": "外层 RAG supervisor 给出的槽位描述",
  "context": "已知事实和运行时提示"
}

# 输出格式
只返回有效 JSON：
{
  "mode": "one | n | existence",
  "rank_order": "top | bottom",
  "limit": 1
}
'''

_llm_interface = LLInterface()
_extractor_llm = LLInterface()
_extractor_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED,
    ),
)


def _normalize_relationship_args(raw_args: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize extractor output into safe relationship-ranking arguments.

    Args:
        raw_args: Parsed JSON object from the relationship extractor LLM.

    Returns:
        Sanitized request dict, or ``None`` when required values are out of schema.
    """
    mode = text_or_empty(raw_args.get("mode"))
    rank_order = text_or_empty(raw_args.get("rank_order"))
    raw_limit = raw_args.get("limit")
    if mode not in _RELATIONSHIP_MODES or rank_order not in _RELATIONSHIP_RANK_ORDERS:
        return None
    if not isinstance(raw_limit, int) or isinstance(raw_limit, bool):
        return None

    limit = max(1, min(raw_limit, _MAX_RELATIONSHIP_LIMIT))
    if mode == "one":
        limit = 1

    return_value = {
        "mode": mode,
        "rank_order": rank_order,
        "limit": limit,
    }
    return return_value


async def _extract_relationship_args(
    task: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract relationship-ranking parameters from the agent task.

    Args:
        task: Relationship slot description produced by the outer supervisor.
        context: Runtime hints passed to the helper agent.

    Returns:
        Sanitized relationship args, or ``None`` when extraction failed.
    """
    system_prompt = SystemMessage(content=_EXTRACTOR_PROMPT)
    llm_context = project_runtime_context_for_llm(context)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": llm_context},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _extractor_llm.ainvoke([system_prompt, human_message], config=_extractor_llm_config)
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return None
    return_value = _normalize_relationship_args(result)
    return return_value


def _relationship_band(label: str) -> str:
    """Classify a public affinity label into a coarse relationship band.

    Args:
        label: Label returned by ``build_affinity_block``.

    Returns:
        ``"positive"``, ``"negative"``, or ``"neutral"``.
    """
    if label in _POSITIVE_LABELS:
        return "positive"
    if label in _NEGATIVE_LABELS:
        return "negative"
    return "neutral"


def _public_candidate(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    """Convert an affinity-ranked profile row into a prompt-safe candidate.

    Args:
        doc: Raw relationship row returned by the DB helper.
        rank: One-based rank after affinity sorting.

    Returns:
        Candidate payload with no raw affinity value.
    """
    affinity_block = build_affinity_block(doc["affinity"])
    label = affinity_block["level"]
    return_value = {
        "rank": rank,
        "global_user_id": doc["global_user_id"],
        "display_name": doc["display_name"],
        "platform": doc["platform"],
        "platform_user_id": doc["platform_user_id"],
        "relationship_label": label,
        "relationship_band": _relationship_band(label),
    }
    return return_value


class RelationshipAgent(BaseRAGHelperAgent):
    """RAG helper agent that ranks users by the character's relationship data.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="relationship_agent",
            cache_name=RELATIONSHIP_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Return prompt-safe relationship candidates for the character.

        Args:
            task: Relationship slot description from the supervisor.
            context: Runtime hints supplying platform scope.
            max_attempts: Unused; kept for interface compatibility.

        Returns:
            Dict with resolved flag, prompt-safe relationship candidates, and
            cache metadata. Raw affinity values are never included.
        """
        del max_attempts

        args = await _extract_relationship_args(task, context)
        if args is None:
            return_value = self.with_cache_status(
                {
                    "resolved": False,
                    "result": "Could not extract relationship ranking parameters from task.",
                    "attempts": 1,
                },
                hit=False,
                reason="skipped_unextractable_relationship_args",
            )
            return return_value

        cache_key = build_relationship_cache_key(args, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return_value = self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )
            return return_value

        platform = str(context.get("platform") or "").strip() or None
        rows = await list_users_by_affinity(
            rank_order=args["rank_order"],
            platform=platform,
            limit=args["limit"],
        )
        candidates = [
            _public_candidate(row, rank)
            for rank, row in enumerate(rows, start=1)
        ]

        result = {
            "mode": args["mode"],
            "rank_order": args["rank_order"],
            "source": "user_profiles_relationship_rank",
            "candidates": candidates,
        }
        if args["mode"] == "existence":
            matching_band = "positive" if args["rank_order"] == "top" else "negative"
            result["has_matching_candidate"] = any(
                candidate["relationship_band"] == matching_band
                for candidate in candidates
            )

        if candidates:
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_relationship_dependencies(context),
                metadata={
                    "mode": args["mode"],
                    "rank_order": args["rank_order"],
                },
            )

        return_value = self.with_cache_status(
            {
                "resolved": bool(candidates),
                "result": result,
                "attempts": 1,
            },
            hit=False,
            reason="miss_stored" if candidates else "miss_unresolved",
            cache_key=cache_key,
        )
        return return_value
