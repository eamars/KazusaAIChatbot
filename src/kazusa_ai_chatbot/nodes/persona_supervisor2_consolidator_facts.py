"""Stage 4 consolidator fact extraction agents."""

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
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import ConsolidatorState
from kazusa_ai_chatbot.utils import get_llm, log_list_preview, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)


_FACTS_HARVESTER_PROMPT = """\
你负责提取具备长期价值的**事实证据**和**未来约定**。这些结果不是最终画像；它们会作为下游 memory-unit consolidator 的证据输入。
你必须严格区分哪些只是“对话复述”（禁止记录），哪些是“以后仍然有用的事实/事件/约定”。

# 背景信息
- **对话主体 (Character)**: {character_name}
- **对话对象 (User)**: human payload 中的 `user_name`
- **系统时间**: human payload 中的 `timestamp`

# 证据分层（必须遵守）
- `decontexualized_input`：用户这一轮真正表达的内容，常包含请求、愿望、试探、调侃。
- `rag_result.user_image.user_memory_context`：RAG 投影给认知层的用户记忆摘要，按 recent_shifts/objective_facts/milestones/stable_patterns/active_commitments 分类。
- `rag_result.user_memory_unit_candidates`：RAG 检索出的原始候选记忆单元，仅用于判断是否旧闻或语义重复。
- `rag_result.memory_evidence` / `conversation_evidence` / `external_evidence`：其他检索证据。
- `content_anchors`：角色在生成回复前的草案意图，只能视为“候选计划”，**不能单独证明承诺已经成立**。
- `final_dialog`：角色本轮最终实际说出口的话，是判断“是否真的接受/承诺/拒绝”的最高优先级证据。
- 当三者冲突时，优先级固定为：`final_dialog` > `content_anchors` > `decontexualized_input`。

# 核心审计准则 (Audit Standards)
1. **身份锚定 [必须执行]**:
   - `decontexualized_input` 的内容始终是 human payload 中 `user_name` 指向的用户在表达。
   - `content_anchors` 的内容始终是 **{character_name}** 在做决定。
   - 严禁出现身份倒置（如：将用户写完作业记在 {character_name} 头上）。

2. **事实证据 (new_facts) 判定标准**:
   - **记录**：以后仍然有用的具体事实、偏好、禁忌、关系声明、重要事件、反复出现的互动模式，或从 `rag_result.external_evidence` 中提取的新信息。
   - **记录粒度**：可以是属性，也可以是带上下文的事件锚点；必须保留足够细节，让下游能写出 fact / subjective_appraisal / relationship_signal。
   - **严禁记录**：纯瞬态动作、空泛情绪、没有后续价值的对话复述，以及任何尚未被角色接下的“奖励”“打算”“计划”。
   - **去重**：如果 `rag_result.user_image.user_memory_context`、`rag_result.user_memory_unit_candidates` 或 `rag_result.memory_evidence` 中已存在相似记忆，严禁重复提取。
   - **硬排除**：`existing_dedup_keys` 是上游给出的已存在事实/承诺键列表；如果候选事实或承诺语义上对应其中任一键，**不要输出**.
   - **语义保真 [必须执行]**：若用户明确说了”喜欢/不喜欢/永远不/一直不/过敏/害怕”等偏好或禁忌，`description` 必须尽量保留原谓词与宾语，不得改写成更宽泛、不同义或模糊的概括。例如”永远不吃辣椒”不能改写为”不喜欢吃杂乱的食物”.
   - **未确认声明 [必须执行]**：当 `logical_stance` 为 `TENTATIVE` 或 `REFUSE`，或 `character_intent` 为 `EVADE` / `REJECT` 时，用户对自身身份、关系或重要属性的任何自我声明（如”我是你学长”、”我们是朋友”）**一律不得落库**——即使改写成”用户自称……”、”用户声称……”等形式也同样禁止。此类输入 `new_facts` 必须返回 `[]`。仅当 `logical_stance` 为 `CONFIRM` 且 `character_intent` 不为 `EVADE` / `REJECT` 时，方可记录用户的身份/关系自我声明.
   - **称呼/句尾/说话格式规则 [必须执行]**：当用户要求 {character_name} 使用特定称呼、句尾、口癖、语气或回复格式（如“主人”“喵”“每句话都这样说”）时，优先判断这是否是一个被角色采纳的**操作性规则/约定**。若角色在 `final_dialog` 中明确接受并准备后续沿用，这类内容应优先进入 `future_promises`（作为持续生效的约定/规则），而不是改写成“{character_name}喜欢/习惯/对这种说话方式感到如何”之类的隐含画像事实。只有当输入本身真的形成了稳定记忆事实时，才可进入 `new_facts`.


3. **承诺 (future_promises) 判定标准 [核心逻辑]**:
   - `future_promises` 记录“**已经被角色采纳的未来义务/约定**”以及“**会持续影响后续回合的操作性规则/接受的指令**”，而不是所有带未来色彩的话题.
   - 先识别 `decontexualized_input` 中的“候选未来事项”，再用 `final_dialog` 判断角色是否真的接下了这件事.
   - **只有当 `final_dialog` 明确体现出接受、答应、确认履行、或形成双方约定时，才能生成 promise。**
   - **只有当 `final_dialog` 明确体现出接受、答应、确认履行、或形成双方约定时，才能生成 promise。**
   - **保留选择权不算承诺 [必须执行]**：若 `logical_stance` 不是 `CONFIRM`，且 `final_dialog` 仍在保留选择权、试探或吊胃口（如“看心情”“谁知道”“到时候再说”“也许吧”“再看”“下次不一定”），`future_promises` 必须返回 `[]`。即使出现“下次”“以后”“回头”等未来词，也不能因为话题带未来性就误判为已接受承诺.
   - 合格条件：表达“谁对谁做什么”的可执行承诺，不是对话复读（如“他说/她问/我觉得”），且能在 `final_dialog` 中找到角色已经接下该义务的证据。
   - 可以接受两种写法：
     1) 不含时间词的承诺本体（推荐）；
     2) 含时间词（如“今晚/明早”）的承诺句，但语义仍是承诺执行动作。
   - 仅在以下情况判 FAIL：
     - 候选 promise 只得到 `decontexualized_input` 或 `content_anchors` 支持，但 `final_dialog` 没有承诺证据；
     - `final_dialog` 表达的是保留选择权、继续试探/调情、模糊敷衍或不确定，而不是接下义务；
     - `action` 是纯计划/猜测（如“可能、也许、打算、准备”）且无明确执行动作；
     - `action` 只是复述对话或主观感受；
     - `action` 与 `target`/基准源主体明显不一致。

# 闭环反馈指南
- 在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)。
- 你需要根据 Evaluator Feedback 对输出做出相应的修正。
- 对未提及内容不要做修改

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "事实所属的主语实名（如 human payload 中的 user_name、{character_name}、或具体物品/地点名）",
            "category": "事实类别标签（如 occupation、location、preference、hobby、relationship、health、schedule、personality 等）",
            "description": "具备长期价值的事实或事件证据。必须包含主语和足够上下文；可以是属性陈述，也可以是具体事件锚点。示例：'用户住在新西兰奥克兰'、'用户在多轮讨论中坚持用系统工程视角校正 Kazusa 记忆架构'。",
            "is_milestone": false,
            "milestone_category": "",
            "scope": "稳定生命周期主题；非里程碑可为空字符串",
            "dedup_key": "稳定去重键"
        }}
    ],
    "future_promises": [
        {{
            "target": "user_name / {character_name}",
            "action": "[姓名]将对[对象]执行[具体动作]（仅承诺本体，不含计划/时间词）",
            "due_time": "ISO 8601 时间戳（如 2026-04-19T06:00:00+12:00），无法确定则为 null",
            "commitment_type": "可选字符串，例如 address_preference / language_preference / future_promise",
            "dedup_key": "稳定承诺更新键"
        }}
    ]
}}
"""
_facts_harvester_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)
async def facts_harvester(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACTS_HARVESTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    rag_result = state["rag_result"]
    msg = {
        "user_name": state["user_name"],
        "timestamp": state["timestamp"],
        "decontexualized_input": state["decontexualized_input"],
        "rag_result": rag_result,
        "supervisor_trace": rag_result.get("supervisor_trace", {}),
        "existing_dedup_keys": sorted(state.get("existing_dedup_keys", set())),
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": state["final_dialog"],
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
        logger.debug(
            "Facts harvester: facts=%d promises=%d facts=%s promises=%s",
            len(new_facts),
            len(future_promises),
            log_list_preview(new_facts),
            log_list_preview(future_promises),
        )

    return {
        "new_facts": new_facts,
        "future_promises": future_promises,
    }


_FACT_HARVESTER_EVALUATOR_PROMPT = """\
你负责审计 Fact Recorder 生成的 JSON 数据。你的核心目标是：**对比“基准源”，核查“候选结果”的准确性和格式合规性。**

# 审计背景
- **角色 (Character)**: {character_name}
- **用户 (User)**: human payload 中的 `user_name`

# 1. 审计基准源 (不可修改的参照物)
- **事实基准**: `decontexualized_input` (仅用于核对 human payload 中 `user_name` 指向用户的状态)
- **承诺基准**: `final_dialog` (角色最终实际说出口的话，优先级最高)
- **承诺辅助基准**: `content_anchors` (仅用于补足 final_dialog 中省略的对象/条件，不能单独制造承诺)
- **历史基准**: `rag_result` (用于检查是否为旧闻)
- **调度轨迹**: `supervisor_trace` (RAG2 loop_count、unknown_slots 与已派发 agent 概览；只能作为检索充分性参考，不能替代事实证据)

# 2. 候选结果 (这是你唯一需要审计的对象)
- **待检事实**: `new_facts`
- **待检承诺**: `future_promises`

# 2.1 与 Harvester 对齐的判定口径（必须遵守）
- `new_facts` 与 `future_promises` 是两个独立通道：
  - `new_facts` 要求“事实/事件证据陈述”，可以是属性，也可以是有上下文的事件锚点。
  - `future_promises.action` 要求“可执行承诺陈述”，**不适用** `new_facts.description` 的属性句式审计规则.
- 当输入没有新增稳定事实时，`new_facts: []` 是合法结果，**不得仅因为空判定失败**.
- 当输入没有明确未来承诺时，`future_promises: []` 也是合法结果.

# 审计红线 (Red Lines)
- **对象倒置**:
  * `decontexualized_input` 里的动作必须记在 human payload 中 `user_name` 指向的用户账上。如果 Recorder 记在 `{character_name}` 头上，立刻拦截.
- **分类错误 [严重]**:
    - 例如：将带有 “未来”、“打算”、“今晚”或任何有许诺时间性质的行为作为 `new_facts`. 这是禁止的行为.
    - 例如：将过去发生的事实存入 `future_promises` 也同样是禁止的行为
- **冗余复读**:
    - 检查候选结果是否只是在复读对话（如“某人问...”).
    - 必须转换为客观陈述.*注意：不要审计输入源的语气，只审计候选结果的陈述方式.*
- **旧闻复读**: 如果该信息在 `rag_result.user_image.user_memory_context`、`rag_result.user_memory_unit_candidates`、`rag_result.memory_evidence` 或 `rag_result.conversation_evidence` 标记的内部库中已存在，判定为 FAIL.
- **脑补事实**: 严禁出现基准源中没有的名词或事实
- **描述格式违规**: `new_facts` 中的 `description` 必须是事实/事件证据陈述，包含明确主语和可核对内容；不能只是情绪标签、泛泛评价或“某人问了一个问题”式复读。
- **类别缺失或不当**: `new_facts` 中的 `category` 必须是有意义的英文标签（如 occupation、location、preference、hobby 等）.若缺失、为空、或为无意义的 "general"，要求补充具体类别.
- **语义漂移 [严重]**: 若 `new_facts.description` 改写后改变了用户原意（尤其是偏好、禁忌、过敏、承诺条件等），必须判 FAIL 并要求使用更贴近原句的表述.
- **未确认声明入库 [严重]**: 若 `logical_stance` 为 `TENTATIVE` 或 `REFUSE`，或 `character_intent` 为 `EVADE` / `REJECT`，而 `new_facts` 中出现了用户对自身身份/关系/属性的自我声明（如"用户是角色的学长"），必须判 FAIL——角色未确认的主张不得作为事实落库.
- **称呼/格式规则通道错误 [严重]**: 若输入核心是用户要求角色采用某种称呼、句尾、口癖、语言或回复格式，而角色在 `final_dialog` 中已经接纳并准备沿用，则优先作为 `future_promises` 中的持续性约定/规则处理，而不是改写成“{character_name}对这种说话方式感到如何”之类的隐含画像事实.
- **承诺 action 审计标准（专用于 `future_promises`）**:
  - 合格条件：表达“谁对谁做什么”的可执行承诺，不是对话复读（如“他说/她问/我觉得”），且能在 `final_dialog` 中找到角色已经接下该义务的证据.
  - 若 `logical_stance` 不是 `CONFIRM`，且 `final_dialog` 仍在保留选择权、试探或吊胃口（如“看心情”“谁知道”“到时候再说”“也许吧”“再看”“下次不一定”），必须判 FAIL 并要求清空 `future_promises`.
  - 可以接受两种写法：
    1) 不含时间词的承诺本体（推荐）；
    2) 含时间词（如“今晚/明早”）的承诺句，但语义仍是承诺执行动作.
  - 仅在以下情况判 FAIL：
    - 候选 promise 只得到 `decontexualized_input` 或 `content_anchors` 支持，但 `final_dialog` 没有承诺证据；
    - `final_dialog` 表达的是保留选择权、继续试探/调情、模糊敷衍或不确定，而不是接下义务；
    - `action` 是纯计划/猜测（如“可能、也许、打算、准备”）且无明确执行动作；
    - `action` 只是复述对话或主观感受；
    - `action` 与 `target`/基准源主体明显不一致.

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_stop": "boolean (如果没有脑补且实名正确且格式合规，返回 true；仅在违反上述明确红线时返回 false。注意：new_facts 为空本身不构成错误。)",
    "feedback": "具体指明错误点。若无实质错误，返回 '通过审计，无需修改'。禁止输出'请确认是否没有新事实'这类非错误性质建议。",
    "contradiction_flags": "可选字符串列表，列举与 rag_result 直接冲突的条目 id 或描述；无冲突则返回 []"
}}
"""
_fact_harvester_evaluator_llm = get_llm(
    temperature=0.1,
    top_p=0.5,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)
async def fact_harvester_evaluator(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACT_HARVESTER_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    retry = state.get("fact_harvester_retry", 0) + 1
    rag_result = state["rag_result"]
    msg = {
        "retry": f"{retry}/{MAX_FACT_HARVESTER_RETRY}",
        "user_name": state["user_name"],
        "new_facts": state["new_facts"],
        "future_promises": state["future_promises"],

        "decontexualized_input": state["decontexualized_input"],
        "rag_result": rag_result,
        "supervisor_trace": rag_result.get("supervisor_trace", {}),
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    response = await _fact_harvester_evaluator_llm.ainvoke([system_prompt, human_message])

    result = parse_llm_json_output(response.content)

    logger.debug(
        "Fact harvester evaluator: retry=%d should_stop=%s contradictions=%s feedback=%s",
        retry,
        result.get("should_stop", True),
        result.get("contradiction_flags", []),
        log_preview(result.get("feedback", "")),
    )

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

    return {
        "should_stop": should_stop,
        "fact_harvester_feedback_message": [feedback_message],
        "fact_harvester_retry": retry,
        "metadata": metadata,
    }
