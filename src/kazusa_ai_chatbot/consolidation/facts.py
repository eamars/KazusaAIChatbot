"""Consolidator fact extraction agents."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.config import MAX_FACT_HARVESTER_RETRY
from kazusa_ai_chatbot.consolidation.origin import (
    project_consolidation_origin_prompt_block,
)
from kazusa_ai_chatbot.consolidation.schema import (
    ConsolidatorState,
    content_anchors_from_action_directives,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.utils import get_llm, log_list_preview, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)


_FACTS_HARVESTER_CANDIDATE_LIMIT = 12
_FACTS_HARVESTER_CANDIDATE_FACT_LIMIT = 240
_FACTS_HARVESTER_CANDIDATE_FIELDS = (
    "unit_id",
    "unit_type",
    "dedup_key",
    "updated_at",
)


def _stripped_candidate_text(value: object) -> str:
    if not isinstance(value, str):
        return_value = ""
        return return_value

    return_value = value.strip()
    return return_value


def _clipped_candidate_fact(value: object) -> str:
    fact = _stripped_candidate_text(value)
    if len(fact) > _FACTS_HARVESTER_CANDIDATE_FACT_LIMIT:
        fact = fact[:_FACTS_HARVESTER_CANDIDATE_FACT_LIMIT].rstrip()

    return_value = fact
    return return_value


def _compact_memory_unit_candidate(candidate: object) -> dict[str, object]:
    """Keep one memory-unit candidate as bounded duplicate evidence.

    Args:
        candidate: Raw surfaced memory-unit row from the global RAG payload.

    Returns:
        A prompt-safe row containing only identity, type, dedup, timestamp, and
        clipped fact text.
    """

    if not isinstance(candidate, dict):
        return_value: dict[str, object] = {}
        return return_value

    selected_candidate: dict[str, object] = {}
    for field in _FACTS_HARVESTER_CANDIDATE_FIELDS + ("fact",):
        if field in candidate:
            selected_candidate[field] = candidate[field]

    projected_candidate = project_tool_result_for_llm(selected_candidate)
    if not isinstance(projected_candidate, dict):
        projected_candidate = selected_candidate

    compact_candidate: dict[str, object] = {}
    for field in _FACTS_HARVESTER_CANDIDATE_FIELDS:
        value = _stripped_candidate_text(projected_candidate.get(field))
        if value:
            compact_candidate[field] = value

    fact = _clipped_candidate_fact(projected_candidate.get("fact"))
    if fact:
        compact_candidate["fact"] = fact

    return_value = compact_candidate
    return return_value


def _compact_memory_unit_candidates(candidates: object) -> list[dict[str, object]]:
    """Build capped memory-unit duplicate hints for facts prompts.

    Args:
        candidates: Raw surfaced memory-unit candidate list from RAG.

    Returns:
        At most `_FACTS_HARVESTER_CANDIDATE_LIMIT` compact candidate rows.
    """

    if not isinstance(candidates, list):
        return_value: list[dict[str, object]] = []
        return return_value

    compact_candidates: list[dict[str, object]] = []
    for candidate in candidates:
        if len(compact_candidates) >= _FACTS_HARVESTER_CANDIDATE_LIMIT:
            break

        compact_candidate = _compact_memory_unit_candidate(candidate)
        if compact_candidate:
            compact_candidates.append(compact_candidate)

    return_value = compact_candidates
    return return_value


def _facts_harvester_rag_view(rag_result: object) -> dict:
    """Return the bounded RAG view used by facts harvester prompts.

    Args:
        rag_result: Global consolidation RAG result, possibly carrying raw
            surfaced memory-unit candidates for later merge reuse.

    Returns:
        Projected RAG evidence with raw memory-unit candidate rows replaced by
        compact prompt-safe duplicate hints.
    """

    if not isinstance(rag_result, dict):
        return_value: dict = {}
        return return_value

    raw_candidates = rag_result.get("user_memory_unit_candidates")
    projected_source = dict(rag_result)
    projected_source.pop("user_memory_unit_candidates", None)

    projected_result = project_tool_result_for_llm(projected_source)
    if not isinstance(projected_result, dict):
        return_value = {}
        return return_value

    projected_result["user_memory_unit_candidates"] = _compact_memory_unit_candidates(
        raw_candidates,
    )
    return_value = projected_result
    return return_value


_FACTS_HARVESTER_PROMPT = '''\
你负责从本轮 consolidation 输入中提取具备长期价值的事实证据 `new_facts` 和已被角色接下的未来约定 `future_promises`。
这些结果不是最终画像；它们会作为下游 memory-unit consolidator 的证据输入。
你必须区分：什么只是对话复述，什么以后仍然有用，什么已经成为 `{character_name}` 的后续义务或持续规则。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的自由文本字段必须使用简体中文。
- `category`、`commitment_type`、`dedup_key` 等机器标签字段按输出格式要求保持英文或稳定键格式。
- `description`、`action` 等语义文本字段使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句只有在必须精确保留时才保持原语言。
- 不添加翻译、双语复写或括号解释，除非源文本本身已经包含。

# 对话与证据身份
- 对话主体是 `{character_name}`。
- 对话对象是 human payload 中的 `user_name`。
- `timestamp` 是当前回合本地时间。
- `consolidation_origin.trigger_source` 指明本轮 consolidation 的来源。
- 当 trigger_source 是 `user_message` 时，`decontexualized_input` 是 `user_name` 指向的用户本轮真正表达的内容。
- 当 trigger_source 是 `internal_thought` 时，`decontexualized_input` 是角色内部触发文本，概括当前认知焦点与证据，不是用户原话。
- `content_anchors` 表示 `{character_name}` 生成回复前的草案意图，只能作为候选计划，不能单独证明承诺已经成立。
- `final_dialog` 是 `{character_name}` 本轮最终输出；`user_message` 时是可见回复，`internal_thought` 时是私有 finalization，是判断接受、拒绝、保留选择权或形成承诺的最高优先级证据。
- `episode_trace_projection` 是动作和表面输出的安全摘要；它只能帮助判断本轮实际选择、执行或跳过了哪些动作，不能替代用户事实来源。
- 如果 `episode_trace_projection.action_results` 显示 `background_artifact_request` 已经进入 `pending`，或 evidence owner 是 `background_artifact_job`，说明稍后交付由后台 artifact job 拥有；不要为同一件后台产物再生成 `future_promises`。它可以作为动作审计证据，但不是 consolidator 需要复制的一条承诺。
- 当 `final_dialog`、`content_anchors`、`decontexualized_input` 冲突时，承诺判断优先级为 `final_dialog` > `content_anchors` > `decontexualized_input`。

# 来源权威性
- 强事实来源：`user_message` 来源中用户在 `decontexualized_input` 明确陈述的自身事实；`rag_result.memory_evidence`、`conversation_evidence`、`external_evidence` 中已有或检索出的事实。`internal_thought` 来源中的 `decontexualized_input` 不是用户事实来源，只能说明当前内部关注点。
- 回忆证据来源：`rag_result.recall_evidence` 可以证明当前约定、承诺、计划或进度的来源。若 `primary_source` 是 `conversation_progress` 且没有 `user_memory_units`、`conversation_history` 或本轮用户明确事实支持，它只能作为回合操作证据，不能单独授权写入 `{character_name}` 的稳定事实。
- 回合局部支持：`final_dialog` 与 `content_anchors` 可说明本轮角色说了什么、准备怎么回应，但不能单独制造角色的长期偏好、角色设定或角色 lore。
- 弱/非事实来源：内部独白、情绪评估、互动潜台词不是客观事实来源；即使上游出现，也只能作为主观体感。
- 生成回复自污染禁止：如果某个候选事实只来自角色本轮即兴回复，而没有用户明确陈述或 `rag_result` 中的检索证据支持，不得写成 `{character_name}` 的稳定偏好、习惯、设定或事实。

# new_facts 判定
- 记录以后仍然有用的具体事实、偏好、禁忌、关系声明、重要事件、反复出现的互动模式，或从 `rag_result.external_evidence` 中提取的新信息。
- `description` 可以是属性陈述，也可以是带上下文的事件锚点；必须包含明确主语和足够细节，让下游能写出 fact、subjective_appraisal、relationship_signal。
- 不记录纯瞬态动作、空泛情绪、没有后续价值的对话复述，以及任何尚未被角色接下的奖励、打算或计划。
- 若候选事实主语是 `{character_name}`，必须先在 `rag_result.memory_evidence`、`conversation_evidence`、`external_evidence`，有 durable/source proof 支持的 `rag_result.recall_evidence`，或用户明确提供的可核对事实中找到非生成证据；找不到时删除。
- 用户向 `{character_name}` 询问偏好、状态或习惯后，`final_dialog` 中的第一人称回答只属于本轮台词，不是稳定事实证据。
- 如果用户明确说了喜欢、不喜欢、永远不、一直不、过敏、害怕等偏好或禁忌，`description` 必须尽量保留原谓词与宾语，不得改写成更宽泛或不同义的概括。
- 当 `logical_stance` 为 `TENTATIVE` 或 `REFUSE`，或 `character_intent` 为 `EVADE` / `REJECT` 时，用户对自身身份、关系或重要属性的自我声明不得写入 `new_facts`，即使改成“用户自称”也不行。
- 如果事实或事件锚点包含日期、时段、截止点或相对时间，能从 `timestamp`、消息时间和可见证据确定具体本地日期时，`description` 使用绝对日期。无法确定且该时间影响事实成立范围时，删除该候选事实。

# future_promises 判定
- `future_promises` 只记录 `{character_name}` 已经接受的未来义务、双方约定、后续行为或会持续影响后续回合的操作性规则。
- 用户要求 `{character_name}` 使用特定称呼、句尾、口癖、语气或回复格式时，如果 `final_dialog` 明确接受并准备后续沿用，优先作为 `future_promises` 中的持续规则，而不是改写成 `{character_name}` 喜欢或习惯这种说法。
- 每个候选 promise 必须通过四步链：
  1. 从 `decontexualized_input` 或 `content_anchors` 识别未来事项或持续规则候选。
  2. 判断现实义务主体是不是 `{character_name}`；用户自己要做、用户计划做、物品流程、当前任务流程、或角色给建议/评价/提醒都不是角色承诺。
  3. 在 `final_dialog` 中找到 `{character_name}` 明确接受、答应、确认后续履行或形成双方约定的证据。
  4. 将 `action` 写成 `{character_name}` 未来要执行或持续遵守的具体动作。
- `action` 只写承诺本体，不写计划词、猜测词、相对时间词、建议、观察、复述、主观感受或用户自己的计划。
- 如果 `final_dialog` 只是建议用户怎么做、认可用户自己的计划、评价方案更稳妥或更合理，`future_promises` 返回空数组。
- 如果 `logical_stance` 不是 `CONFIRM`，且 `final_dialog` 仍在保留选择权、试探或吊胃口，不能因为话题有未来色彩就输出 promise。
- 如果承诺有可确定的执行时间、截止时间、展示日、挑战日、提醒日或验收日，写入 `due_time`，格式为本地 `YYYY-MM-DD HH:MM`；只有日期没有时刻时使用该本地日期的 `00:00`。
- 无到期日的持续规则可以使用 `due_time: null`。
- 如果承诺是否成立依赖 `今天`、`今晚`、`明天`、`明早`、`之后`、`下次`、`later`、`next time` 等相对时间或相对顺序，而你无法从 `timestamp`、消息时间和可见证据解析出具体日期、时间或当前状态，不输出该候选 promise。

# 去重与硬排除
- 如果 `rag_result.user_image.user_memory_context`、`rag_result.user_memory_unit_candidates` 或 `rag_result.memory_evidence` 中已存在相似记忆，严禁重复提取。
- `existing_dedup_keys` 是上游给出的已存在事实或承诺键列表；如果候选事实或承诺语义上对应其中任一键，不要输出。

# 闭环反馈
- 生成前检查输入消息列表中最后一条来自 Evaluator 的反馈。
- 只修正反馈提到的问题；不要为了迎合反馈新增没有证据的内容。

# 生成步骤
1. 读取 `user_name`、`timestamp`、`decontexualized_input`，确认本轮是谁在表达、表达了什么。
2. 读取 `final_dialog`，判断 `{character_name}` 最终是否接受、拒绝、保留选择权或形成承诺。
3. 检查 `rag_result.user_image.user_memory_context`、`rag_result.user_memory_unit_candidates`、`rag_result.memory_evidence`、`rag_result.recall_evidence` 与 `existing_dedup_keys`，过滤旧闻和重复内容。
4. 对事实候选执行“来源权威性 -> 长期价值 -> 主体正确 -> 时间有效性”检查，合格才写入 `new_facts`。
5. 对承诺候选执行“未来事项 -> 义务主体 -> final_dialog 接受证据 -> action 可执行性 -> due_time 写法”检查，合格才写入 `future_promises`。
6. 对 `{character_name}` 自身事实候选执行非生成证据检查；没有非生成证据时删除。
7. 没有合格事实或承诺时，对应字段返回空数组；不要为了填满输出而复述对话。

# 输入格式
human payload 是以下 JSON：
{{
    "user_name": "当前用户显示名",
    "timestamp": "当前回合本地时间，YYYY-MM-DD HH:MM",
    "consolidation_origin": {{
        "episode_id": "string",
        "trigger_source": "user_message | internal_thought",
        "input_sources": ["..."],
        "output_mode": "string"
    }},
    "decontexualized_input": "用户本轮真实意图摘要",
    "rag_result": {{
        "user_image": {{
            "user_memory_context": {{
                "stable_patterns": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间"}}],
                "recent_shifts": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间"}}],
                "objective_facts": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间"}}],
                "milestones": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间"}}],
                "active_commitments": [{{"fact": "...", "subjective_appraisal": "...", "relationship_signal": "...", "updated_at": "本地时间"}}]
            }}
        }},
        "user_memory_unit_candidates": ["已压缩的候选记忆单元，仅含 unit_id/unit_type/fact/dedup_key/updated_at"],
        "memory_evidence": ["相关长期记忆证据"],
        "recall_evidence": ["约定/承诺/进度回忆证据"],
        "conversation_evidence": ["相关近期对话证据"],
        "external_evidence": ["相关外部证据"],
        "supervisor_trace": {{"unknown_slots": ["未解决槽位"], "loop_count": 1}}
    }},
    "supervisor_trace": {{"unknown_slots": ["未解决槽位"], "loop_count": 1}},
    "existing_dedup_keys": ["已存在事实或承诺的稳定去重键"],
    "content_anchors": ["回复前的内容锚点"],
    "final_dialog": ["{character_name} 本轮可见回复或私有 finalization"],
    "episode_trace_projection": {{
        "action_results": ["动作结果摘要，不包含 handler、raw params 或数据库内部字段"],
        "surface_outputs": ["表面输出摘要"]
    }},
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "事实所属的主语实名（如 human payload 中的 user_name、{character_name}、或具体物品/地点名）",
            "category": "事实类别标签（如 occupation、location、preference、hobby、relationship、health、schedule、personality 等）",
            "description": "具备长期价值的事实或事件证据；必须包含主语和足够上下文",
            "is_milestone": false,
            "milestone_category": "",
            "scope": "稳定生命周期主题；非里程碑可为空字符串",
            "dedup_key": "稳定去重键"
        }}
    ],
    "future_promises": [
        {{
            "target": "user_name / {character_name}",
            "action": "[姓名]将对[对象]执行[具体动作]（仅承诺本体，不含计划词或相对时间词）",
            "due_time": "本地 YYYY-MM-DD HH:MM；无到期日的持续规则为 null；无法解析相对时间的时间性承诺不要输出",
            "commitment_type": "可选字符串，例如 address_preference / language_preference / future_promise",
            "dedup_key": "稳定承诺更新键"
        }}
    ]
}}
'''
_facts_harvester_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)
async def facts_harvester(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(content=_FACTS_HARVESTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    local_datetime = state["local_time_context"]["current_local_datetime"]
    rag_result = _facts_harvester_rag_view(state["rag_result"])
    msg = {
        "user_name": state["user_name"],
        "timestamp": local_datetime,
        "consolidation_origin": project_consolidation_origin_prompt_block(
            state["consolidation_origin"]
        ),
        "decontexualized_input": state["decontexualized_input"],
        "rag_result": rag_result,
        "supervisor_trace": rag_result.get("supervisor_trace", {}),
        "existing_dedup_keys": sorted(state.get("existing_dedup_keys", set())),
        "content_anchors": content_anchors_from_action_directives(
            state.get("action_directives"),
        ),
        "final_dialog": state["final_dialog"],
        "episode_trace_projection": state.get("episode_trace_projection", {}),
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    # Trim evaluator feedback to the first + latest three messages.
    feedback = state.get("fact_harvester_feedback_message", []) or []
    if len(feedback) > 3:
        recent_messages = [feedback[0]] + feedback[-3:]
    else:
        recent_messages = feedback

    response = await _facts_harvester_llm.ainvoke([system_prompt, human_message] + recent_messages)

    result = parse_llm_json_output(response.content)

    new_facts = result.get("new_facts", []) or []
    future_promises = result.get("future_promises", []) or []
    if new_facts or future_promises:
        logger.debug(f'Facts harvester: facts={len(new_facts)} promises={len(future_promises)} facts={log_list_preview(new_facts)} promises={log_list_preview(future_promises)}')

    return_value = {
        "new_facts": new_facts,
        "future_promises": future_promises,
    }
    return return_value


_FACT_HARVESTER_EVALUATOR_PROMPT = '''\
你负责审计 Fact Recorder 生成的 JSON 数据。你的目标是对照基准源，判断候选 `new_facts` 和 `future_promises` 是否准确、分流正确、格式合规。
你不是新的事实生成器；只在发现明确红线时要求修正。

# 语言政策
- 除结构化枚举值、schema key、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的自由文本字段必须使用简体中文。
- 候选结果中的机器标签字段按原样审计，不翻译 schema key、枚举值、ID 或 dedup key。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句只有在必须精确保留时才保持原语言。
- 不添加翻译、双语复写或括号解释，除非源文本本身已经包含。

# 审计身份与基准源
- 角色是 `{character_name}`。
- 用户是 human payload 中的 `user_name`。
- `consolidation_origin.trigger_source` 指明本轮 consolidation 的来源。
- 当 trigger_source 是 `user_message` 时，`decontexualized_input` 是用户本轮表达，可用于核对 `user_name` 指向用户的状态、事实或偏好。
- 当 trigger_source 是 `internal_thought` 时，`decontexualized_input` 是角色内部触发文本，不是用户原话；用户事实必须来自 `rag_result`、近期对话证据或其他明确证据。
- 承诺基准是 `final_dialog`；`user_message` 时它是角色可见回复，`internal_thought` 时它是私有 finalization。
- `content_anchors` 只能补足 `final_dialog` 中省略的对象或条件，不能单独制造承诺。
- `rag_result` 用于检查旧闻、重复和非生成证据。
- `supervisor_trace` 只能作为检索充分性参考，不能替代事实或承诺证据。
- `episode_trace_projection` 是动作和表面输出的安全摘要；它只能帮助判断本轮实际选择、执行或跳过了哪些动作，不能替代用户事实来源。
- 如果候选 `future_promises` 只是重复一个已由 `background_artifact_request` / `background_artifact_job` 拥有的稍后产物交付，必须判 FAIL 并要求删除该候选；后台 job 已经是操作性所有者。

# 来源权威性审计
- 强事实来源：`user_message` 来源中用户在 `decontexualized_input` 明确陈述的自身事实，以及 `rag_result.memory_evidence`、`conversation_evidence`、`external_evidence` 中的证据。`internal_thought` 来源中的 `decontexualized_input` 不是用户事实来源。
- 回忆证据来源：`rag_result.recall_evidence` 是约定、承诺或进度的来源证据。progress-only recall 只能说明当前操作状态；若没有 durable memory、conversation proof 或本轮用户明确事实支持，不能作为 `{character_name}` 稳定事实依据。
- 回合局部支持：`final_dialog` 与 `content_anchors` 只能证明本轮说法或候选计划，不能单独制造角色长期偏好、角色设定或角色 lore。
- 弱/非事实来源：内部独白、情绪评估、互动潜台词不是客观事实来源。
- 若 `new_facts` 中的 `{character_name}` 稳定事实只来自 generated dialog 或 `final_dialog`，而没有 `rag_result` 中的检索证据或用户明确事实支持，必须判 FAIL。
- 若用户只是询问 `{character_name}` 的偏好、状态或习惯，而候选事实来自角色在 `final_dialog` 中的第一人称回答，必须判 FAIL。

# 候选通道审计
- `new_facts` 要求事实或事件证据陈述，可以是属性，也可以是有上下文的事件锚点。
- `future_promises.action` 要求可执行承诺陈述，不适用 `new_facts.description` 的属性句式审计规则。
- 输入没有新增稳定事实时，`new_facts: []` 合法，不得仅因为空而失败。
- 输入没有明确未来承诺时，`future_promises: []` 合法。

# new_facts 红线
- 对象倒置：`decontexualized_input` 里的用户动作、状态或偏好不能记在 `{character_name}` 头上。
- 分类错误：尚未被角色接下的未来奖励、打算、计划、提醒、截止事项不能作为 `new_facts` 保存。
- 冗余复读：候选不能只是“某人问了什么”“某人说了什么”式对话复述；必须是客观事实或事件证据。
- 旧闻复读：如果该信息在 `rag_result.user_image.user_memory_context`、`rag_result.user_memory_unit_candidates`、`rag_result.memory_evidence` 或 `rag_result.conversation_evidence` 中已存在，判 FAIL。
- 脑补事实：候选不能出现基准源中没有的名词、事实、偏好、禁忌或关系。
- 描述格式违规：`description` 必须包含明确主语和可核对内容，不能只是情绪标签或泛泛评价。
- 类别缺失或不当：`category` 必须是有意义的英文标签；缺失、为空或无意义的 `general` 要求修正。
- 语义漂移：若 `description` 改写后改变用户原意，尤其是偏好、禁忌、过敏、承诺条件，必须判 FAIL。
- 未确认声明入库：若 `logical_stance` 为 `TENTATIVE` 或 `REFUSE`，或 `character_intent` 为 `EVADE` / `REJECT`，用户对自身身份、关系或重要属性的自我声明不得作为事实落库。
- 称呼或格式规则通道错误：若用户要求角色采用某种称呼、句尾、口癖、语言或回复格式，且角色在 `final_dialog` 中已经接纳并准备沿用，应作为 `future_promises` 的持续规则，而不是写成 `{character_name}` 的隐含画像事实。
- 时间性事实审计：如果 `description` 使用 `今天`、`今晚`、`明天`、`之后`、`下次`、`later`、`next time` 等相对时间来表达稳定事实或事件锚点，且没有写出可从基准源核对的绝对日期或当前状态，必须判 FAIL。

# future_promises 红线
- 每个候选承诺必须通过四步链：候选未来事项 -> 义务主体 -> `final_dialog` 接受证据 -> `action` 可执行性。任一步失败必须判 FAIL。
- `action` 必须表达 `{character_name}` 未来要做或持续遵守什么，不能是用户自己的计划、物品流程、当前任务流程、建议、观察、复述或主观感受。
- 如果 `final_dialog` 只是建议用户怎么做、认可用户自己的计划、评价方案更稳妥或更合理，必须判 FAIL 并要求清空对应 `future_promises`。
- 若 `logical_stance` 不是 `CONFIRM`，且 `final_dialog` 仍在保留选择权、试探或吊胃口，必须判 FAIL。
- `action` 不得包含计划词、猜测词或相对时间词；到期或执行时间只应出现在 `due_time`。
- 如果候选承诺有执行时间、截止时间、提醒时间、展示日、挑战日或验收日，`due_time` 必须是本地 `YYYY-MM-DD HH:MM`。
- 无到期日的持续规则可以 `due_time: null`。
- 如果候选承诺依赖相对时间或相对顺序才能成立，但 `due_time` 为 null，且无法从 `decontexualized_input`、`content_anchors`、`final_dialog`、`timestamp` 或消息证据看出绝对日期、时间或当前状态，必须判 FAIL 并要求删除该候选。

# 审计步骤
1. 确认 `user_name` 和 `{character_name}`，逐项检查 `new_facts` 与 `future_promises` 是否主体倒置。
2. 用 `decontexualized_input` 核对事实候选，用 `final_dialog` 核对承诺候选。
3. 用 `rag_result.user_image.user_memory_context`、`rag_result.user_memory_unit_candidates`、`rag_result.memory_evidence` 与 `conversation_evidence` 检查旧闻复读。
4. 检查 `logical_stance` 和 `character_intent` 是否允许记录用户自我声明或未来承诺。
5. 分别按 `new_facts` 红线和 `future_promises` 红线审计时间表达、通道分流和格式。
6. 只有发现明确红线时才返回 `should_stop: false`；空数组本身不是错误。

# 输入格式
human payload 是以下 JSON：
{{
    "retry": "当前重试次数/最大重试次数",
    "user_name": "当前用户显示名",
    "consolidation_origin": {{
        "episode_id": "string",
        "trigger_source": "user_message | internal_thought",
        "input_sources": ["..."],
        "output_mode": "string"
    }},
    "new_facts": [
        {{
            "entity": "事实所属主语",
            "category": "事实类别标签",
            "description": "事实或事件证据",
            "is_milestone": false,
            "milestone_category": "",
            "scope": "",
            "dedup_key": "稳定去重键"
        }}
    ],
    "future_promises": [
        {{
            "target": "承诺目标",
            "action": "可执行承诺本体，不含计划词或相对时间词",
            "due_time": "本地 YYYY-MM-DD HH:MM，或无到期日持续规则的 null",
            "commitment_type": "承诺类型",
            "dedup_key": "稳定承诺更新键"
        }}
    ],
    "decontexualized_input": "用户本轮真实意图摘要",
    "rag_result": {{
        "user_image": {{"user_memory_context": "五类用户记忆单元投影"}},
        "user_memory_unit_candidates": ["已压缩的候选记忆单元，仅含 unit_id/unit_type/fact/dedup_key/updated_at"],
        "memory_evidence": ["相关长期记忆证据"],
        "recall_evidence": ["约定/承诺/进度回忆证据"],
        "conversation_evidence": ["相关近期对话证据"],
        "external_evidence": ["相关外部证据"],
        "supervisor_trace": {{"unknown_slots": ["未解决槽位"], "loop_count": 1}}
    }},
    "supervisor_trace": {{"unknown_slots": ["未解决槽位"], "loop_count": 1}},
    "episode_trace_projection": {{
        "action_results": ["安全动作结果摘要"],
        "surface_outputs": ["安全表面输出摘要"]
    }},
    "content_anchors": ["回复前的内容锚点"],
    "final_dialog": ["{character_name} 本轮可见回复或私有 finalization"],
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_stop": "boolean；没有脑补、主体正确且格式合规时返回 true；仅在违反明确红线时返回 false",
    "feedback": "具体指明错误点；若无实质错误，返回 '通过审计，无需修改'；禁止输出非错误性质建议",
    "contradiction_flags": "可选字符串列表，列举与 rag_result 直接冲突的条目 id 或描述；无冲突则返回 []"
}}
'''
_fact_harvester_evaluator_llm = get_llm(
    temperature=0.1,
    top_p=0.5,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)
async def fact_harvester_evaluator(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(content=_FACT_HARVESTER_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    retry = state.get("fact_harvester_retry", 0) + 1
    rag_result = _facts_harvester_rag_view(state["rag_result"])
    msg = {
        "retry": f"{retry}/{MAX_FACT_HARVESTER_RETRY}",
        "user_name": state["user_name"],
        "consolidation_origin": project_consolidation_origin_prompt_block(
            state["consolidation_origin"]
        ),
        "new_facts": state["new_facts"],
        "future_promises": state["future_promises"],

        "decontexualized_input": state["decontexualized_input"],
        "rag_result": rag_result,
        "supervisor_trace": rag_result.get("supervisor_trace", {}),
        "episode_trace_projection": state.get("episode_trace_projection", {}),
        "content_anchors": content_anchors_from_action_directives(
            state.get("action_directives"),
        ),
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _fact_harvester_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(f'Fact harvester evaluator: retry={retry} should_stop={result.get("should_stop", True)} contradictions={result.get("contradiction_flags", [])} feedback={log_preview(result.get("feedback", ""))}')

    should_stop = result.get("should_stop", True)
    if retry >= MAX_FACT_HARVESTER_RETRY:
        should_stop = True

    feedback_message = HumanMessage(
        content=json.dumps(
            {
                "feedback": result.get("feedback", "No feedback"),
                "source": "evaluator",
            },
            ensure_ascii=False,
        ),
        name="evaluator",
    )

    # Propagate evaluator metadata so db_writer can see contradiction flags + retry count.
    contradiction_flags = result.get("contradiction_flags") or []
    existing_meta = state.get("metadata", {}) or {}
    metadata = {
        **existing_meta,
        "fact_harvester_retry": retry,
        "evaluator_passes": existing_meta.get("evaluator_passes", 0) + 1,
        "contradiction_flags": contradiction_flags,
    }

    return_value = {
        "should_stop": should_stop,
        "fact_harvester_feedback_message": [feedback_message],
        "fact_harvester_retry": retry,
        "metadata": metadata,
    }
    return return_value
