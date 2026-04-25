"""Stage 4 consolidator fact extraction agents."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import MAX_FACT_HARVESTER_RETRY
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_schema import ConsolidatorState
from kazusa_ai_chatbot.utils import get_llm, log_list_preview, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)


_FACTS_HARVESTER_PROMPT = """\
你负责提取具备长期价值的**画像属性**（事实）和**未来约定**（承诺）。你必须严格区分哪些是“对话的复述”（禁止记录），哪些是“状态的改变”。

# 背景信息
- **对话主体 (Character)**: {character_name}
- **对话对象 (User)**: {user_name}
- **系统时间**: {timestamp}

# 证据分层（必须遵守）
- `decontexualized_input`：用户这一轮真正表达的内容，常包含请求、愿望、试探、调侃。
- `content_anchors`：角色在生成回复前的草案意图，只能视为“候选计划”，**不能单独证明承诺已经成立**。
- `final_dialog`：角色本轮最终实际说出口的话，是判断“是否真的接受/承诺/拒绝”的最高优先级证据。
- 当三者冲突时，优先级固定为：`final_dialog` > `content_anchors` > `decontexualized_input`。

# 核心审计准则 (Audit Standards)
1. **身份锚定 [必须执行]**:
   - `decontexualized_input` 的内容始终是 **{user_name}** 在表达。
   - `content_anchors` 的内容始终是 **{character_name}** 在做决定。
   - 严禁出现身份倒置（如：将 {user_name} 写完作业记在 {character_name} 头上）。

2. **事实 (new_facts) 判定标准**:
   - **记录**：具有长期稳定性的**属性级陈述**（如：{user_name}的职业、住址、对某物的长期厌恶/偏好）。
   - **记录**：从 `research_facts.external_rag_results` 中提取的**新**信息。
   - **严禁记录**：瞬态动作、对话内容、以及任何关于”奖励”、”打算”、”计划”的内容。
   - **去重**：如果 `research_facts.user_image` 或 `research_facts.input_context_results` 中已存在相似画像，严禁重复提取。
   - **语义保真 [必须执行]**：若用户明确说了”喜欢/不喜欢/永远不/一直不/过敏/害怕”等偏好或禁忌，`description` 必须尽量保留原谓词与宾语，不得改写成更宽泛、不同义或模糊的概括。例如”永远不吃辣椒”不能改写为”不喜欢吃杂乱的食物”。
   - **未确认声明 [必须执行]**：当 `logical_stance` 为 `TENTATIVE` 或 `DENY`，或 `character_intent` 为 `EVADE` / `DENY` 时，用户对自身身份、关系或重要属性的任何自我声明（如”我是你学长”、”我们是朋友”）**一律不得落库**——即使改写成”用户自称……”、”用户声称……”等形式也同样禁止。此类输入 `new_facts` 必须返回 `[]`。仅当 `logical_stance` 为 `CONFIRM` 且 `character_intent` 不为 `EVADE` / `DENY` 时，方可记录用户的身份/关系自我声明。
   - **称呼/句尾/说话格式规则 [必须执行]**：当用户要求 {character_name} 使用特定称呼、句尾、口癖、语气或回复格式（如“主人”“喵”“每句话都这样说”）时，优先判断这是否是一个被角色采纳的**操作性规则/约定**。若角色在 `final_dialog` 中明确接受并准备后续沿用，这类内容应优先进入 `future_promises`（作为持续生效的约定/规则），而不是改写成“{character_name}喜欢/习惯/对这种说话方式感到如何”之类的隐含画像事实。只有当输入本身真的形成了稳定画像属性时，才可进入 `new_facts`。

3. **承诺 (future_promises) 判定标准 [核心逻辑]**:
   - `future_promises` 记录“**已经被角色采纳的未来义务/约定**”以及“**会持续影响后续回合的操作性规则/接受的指令**”，而不是所有带未来色彩的话题。
   - 先识别 `decontexualized_input` 中的“候选未来事项”，再用 `final_dialog` 判断角色是否真的接下了这件事。
   - **只有当 `final_dialog` 明确体现出接受、答应、确认履行、或形成双方约定时，才能生成 promise。**
   - `content_anchors` 仅用于补足 `final_dialog` 里省略的主语、对象、触发条件；**不得在 `final_dialog` 没有承诺证据时单独创造 promise。**
   - 用户的请求、愿望、挑逗、建议、命令、试探，**本身不是 promise**。如果角色最终只是敷衍、保留选择权、继续调情、表达不确定，返回 `future_promises: []`。
   - 若用户要求的是持续性回复规则（如特定称呼、句尾、语言、格式），而角色在 `final_dialog` 中明确接纳并准备沿用，可将其作为**持续性约定**写入 `future_promises`；此时 `due_time` 可为 `null`。
   - 必须包含：[触发条件] + [谁对谁做] + [具体动作]。
   - 承诺时间计算：若角色提出，或者答应了一个将来会发生的事件，则需要根据当前时间推算 `due_time`。
   - `due_time` 推算规则：若输入出现明确时间线索（如“今晚/明早/明天早上/下周末”），必须换算为 ISO 8601 时间戳；仅在完全缺失时间线索时才允许 `null`。
   - **硬约束**：当 `decontexualized_input` 或 `content_anchors` 含有“今晚/晚上/明天/明早/早上/下午/下周”等时间词时，`due_time` 禁止为 `null`。
   - 相对时间默认映射（用于降低歧义）：
     - “今晚/晚上” -> 当日 21:00（若当前已过 21:00，则次日 21:00）
     - “明早/明天早上” -> 次日 08:00
     - “明天下午” -> 次日 15:00
     - “明天晚上” -> 次日 21:00
   - 计算时区使用系统时间 `{timestamp}` 的本地时区；输出必须为 ISO 8601。
   - 若 `decontexualized_input` 是用户对未来行为的请求（如“晚上奖励我”），只有在 `final_dialog` 中被角色明确接纳时，才视为可落库的“待履行承诺”。
   - 若出现冲突信号（如 `content_anchors` 倾向接受，但 `final_dialog` 保留选择权或明显拒绝），以 `final_dialog` 为准并不记录。
   - 对“软性答应/模糊同意”，只有在 `final_dialog` 仍然体现出角色已经接下义务时才可记录；若 `final_dialog` 只是延续氛围、试探、调侃或保留决定权，则不记录。
   - 若 `logical_stance=TENTATIVE` 或 `character_intent=BANTAR/EVADE`，且 `final_dialog` 出现“看心情/谁知道/不保证/也许/可能/下次再说”等保留决定权信号，必须返回 `future_promises: []`。
   - `action` 只写“可执行承诺本体”，不得写时间推测或计划态词汇。**禁止出现**：`计划`、`打算`、`准备`、`想要`、`可能`、`也许`、`明天`、`今晚`、`下次` 等时间/意图词。
   - 时间信息统一写入 `due_time`；若无法确定具体时间可为 `null`，但不要把时间短语塞进 `action`。
   - `action` 必须是去叙事的承诺句，推荐模板：`[执行者]在[触发条件]对[对象]执行[动作]` 或 `[执行者]将对[对象]执行[动作]`。
   - `action` 的触发条件允许写“完成作业后/满足约定后”等条件，但不允许写具体时间点。
   - 若同时存在“触发条件 + 时间线索”，两者都保留：触发条件留在 `action`，时间线索写入 `due_time`。
   - 只有在“纯提问且无任何认可信号（含 logical_stance 与 content_anchors）”时，才不要生成 promise。
   - 所有承诺都填写 `dedup_key`：代表同一承诺或持续规则的稳定英文或原文短键，后续更新同一承诺时必须使用同一个键。

4. **里程碑标记 (is_milestone)**:
   - 若事实属于以下类别之一，则将 `is_milestone` 置为 `true`，否则为 `false`：
     - `preference`（明确偏好）: 用户明确、持久表达”喜欢/讨厌/永远不/一直”等偏好陈述（如”{user_name}永远不吃辣”）。
     - `relationship_state`（关系状态）: 双方关系发生明确变化（如”我们现在是朋友了”、”我信任{character_name}”）。
     - `permission`（许可/禁止）: 用户对 {character_name} 授权或限制的明确声明（如”你可以叫我小名”、”不许再提这件事”）。
     - `revelation`（重大披露）: 用户首次透露的重要身份/健康/隐私信息（如职业、重大疾病、秘密）。
   - 命中任意类别时，`milestone_category` 填写对应英文键（`preference` / `relationship_state` / `permission` / `revelation`）。
   - 普通事实（临时状态、日常行程等）设 `is_milestone: false`，`milestone_category: “”`。
   - 若 `is_milestone=true`，必须填写 `scope`：代表同一生命周期主题的稳定英文键，例如 `language_preference`、`relationship_state`、`health_disclosure`。只有同一 `scope` 的新里程碑会替代旧里程碑。
   - 所有事实都填写 `dedup_key`：用于去重的稳定英文或原文短键，语义相同的事实必须使用同一个键。

5. **future_promises 示例（必须遵守）**:
   - ✅ 正确：
     - `action`: `杏山千纱在EAMARS完成作业后对EAMARS执行奖励`
     - `due_time`: `2026-04-19T06:00:00+12:00`
   - ❌ 错误：
     - `action`: `杏山千纱今晚对EAMARS执行奖励`
     - `action`: `杏山千纱计划于明天早上奖励EAMARS`
     - `action`: `杏山千纱打算下次奖励EAMARS`

6. **拒绝复读**:
   - 严禁记录”某人说了某话”。如果信息已经由对话历史承载，且不涉及长期画像更新，则返回空列表。

# 闭环反馈指南
- 在生成回复前，请检查输入信息列表中的最后一条来自 Evaluator 的消息 (Evaluator Feedback)。
- 你需要根据 Evaluator Feedback 对输出做出相应的修正。
- 对未提及内容不要做修改

# RAG 元信息（仅供参考）
- `rag_metadata` 提供了上游 RAG 的 cache_hit / depth / confidence 等信号。当 `cache_hit=true` 或 `depth=SHALLOW` 时，`research_facts` 的覆盖面可能有限，谨慎判断是否“已知画像”。

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "new_facts": [
        {{
            "entity": "事实所属的主语实名（如 {user_name}、{character_name}、或具体物品/地点名）",
            "category": "事实类别标签（如 occupation、location、preference、hobby、relationship、health、schedule、personality 等）",
            "description": "属性级的客观陈述。格式：'[主语] + [属性/状态]'。示例：'{user_name}住在新西兰奥克兰' 或 '{user_name}对猫毛过敏'。严禁使用叙事句式（如'在某情境下做了某事'）。",
            "is_milestone": false,
            "milestone_category": "",
            "scope": "稳定生命周期主题；非里程碑可为空字符串",
            "dedup_key": "稳定去重键"
        }}
    ],
    "future_promises": [
        {{
            "target": "{user_name} / {character_name}",
            "action": "[姓名]将对[对象]执行[具体动作]（仅承诺本体，不含计划/时间词）",
            "due_time": "ISO 8601 时间戳（如 2026-04-19T06:00:00+12:00），无法确定则为 null",
            "commitment_type": "可选字符串，例如 address_preference / language_preference / future_promise",
            "dedup_key": "稳定承诺更新键"
        }}
    ]
}}
"""
_facts_harvester_llm = get_llm(temperature=0.0, top_p=1.0)
async def facts_harvester(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACTS_HARVESTER_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
        timestamp=state["timestamp"],
    ))

    metadata = state.get("metadata", {}) or {}
    msg = {
        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
        "content_anchors": state["action_directives"]["linguistic_directives"]["content_anchors"],
        "final_dialog": state["final_dialog"],
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "rag_metadata": {
            "cache_hit": metadata.get("cache_hit", False),
            "depth": metadata.get("depth", "DEEP"),
            "depth_confidence": metadata.get("depth_confidence", 0.0),
            "rag_sources_used": metadata.get("rag_sources_used", []),
            "response_confidence": metadata.get("response_confidence", {}),
        },
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

    logger.debug(
        "Facts harvester: facts=%d promises=%d fact_preview=%s promise_preview=%s",
        len(result.get("new_facts", []) or []),
        len(result.get("future_promises", []) or []),
        log_list_preview(result.get("new_facts", []) or []),
        log_list_preview(result.get("future_promises", []) or []),
    )

    return {
        "new_facts": result.get("new_facts", []),
        "future_promises": result.get("future_promises", []),
    }


_FACT_HARVESTER_EVALUATOR_PROMPT = """\
你负责审计 Fact Recorder 生成的 JSON 数据。你的核心目标是：**对比“基准源”，核查“候选结果”的准确性和格式合规性。**

# 审计背景
- **角色 (Character)**: {character_name}
- **用户 (User)**: {user_name}

# 1. 审计基准源 (不可修改的参照物)
- **事实基准**: `decontexualized_input` (仅用于核对 {user_name} 的状态)
- **承诺基准**: `final_dialog` (角色最终实际说出口的话，优先级最高)
- **承诺辅助基准**: `content_anchors` (仅用于补足 final_dialog 中省略的对象/条件，不能单独制造承诺)
- **历史基准**: `research_facts` (用于检查是否为旧闻)

# 2. 候选结果 (这是你唯一需要审计的对象)
- **待检事实**: `new_facts`
- **待检承诺**: `future_promises`

# 2.1 与 Harvester 对齐的判定口径（必须遵守）
- `new_facts` 与 `future_promises` 是两个独立通道：
  - `new_facts` 要求“属性级事实陈述”。
  - `future_promises.action` 要求“可执行承诺陈述”，**不适用** `new_facts.description` 的属性句式审计规则。
- 当输入没有新增稳定事实时，`new_facts: []` 是合法结果，**不得仅因为空判定失败**。
- 当输入没有明确未来承诺时，`future_promises: []` 也是合法结果。

# 审计红线 (Red Lines)
- **对象倒置**:
  * `decontexualized_input` 里的动作必须记在 `{user_name}` 账上。如果 Recorder 记在 `{character_name}` 头上，立刻拦截。
- **分类错误 [严重]**:
    - 例如：将带有 “未来”、“打算”、“今晚”或任何有许诺时间性质的行为作为 `new_facts`。这是禁止的行为。
    - 例如：将过去发生的事实存入 `future_promises` 也同样是禁止的行为
- **冗余复读**:
    - 检查候选结果是否只是在复读对话（如“某人问...”）。
    - 必须转换为客观陈述。*注意：不要审计输入源的语气，只审计候选结果的陈述方式。*
- **旧闻复读**: 如果该信息在 `research_facts.user_image` 或 `research_facts.input_context_results` 标记的内部库中已存在，判定为 FAIL。
- **脑补事实**: 严禁出现基准源中没有的名词或事实
- **描述格式违规**: `new_facts` 中的 `description` 必须是属性级陈述（如 '{user_name}住在奥克兰'），严禁使用叙事句式（如 '在某情境下做了某事'）。若发现叙事句式，要求改写为属性陈述。
- **类别缺失或不当**: `new_facts` 中的 `category` 必须是有意义的英文标签（如 occupation、location、preference、hobby 等）。若缺失、为空、或为无意义的 "general"，要求补充具体类别。
- **语义漂移 [严重]**: 若 `new_facts.description` 改写后改变了用户原意（尤其是偏好、禁忌、过敏、承诺条件等），必须判 FAIL 并要求使用更贴近原句的表述。
- **未确认声明入库 [严重]**: 若 `logical_stance` 为 `TENTATIVE` 或 `DENY`，或 `character_intent` 为 `EVADE` / `DENY`，而 `new_facts` 中出现了用户对自身身份/关系/属性的自我声明（如"用户是角色的学长"），必须判 FAIL——角色未确认的主张不得作为事实落库。
- **称呼/格式规则通道错误 [严重]**: 若输入核心是用户要求角色采用某种称呼、句尾、口癖、语言或回复格式，而角色在 `final_dialog` 中已经接纳并准备沿用，则优先作为 `future_promises` 中的持续性约定/规则处理，而不是改写成“{character_name}对这种说话方式感到如何”之类的隐含画像事实。
- **承诺 action 审计标准（专用于 `future_promises`）**:
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

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "should_stop": "boolean (如果没有脑补且实名正确且格式合规，返回 true；仅在违反上述明确红线时返回 false。注意：new_facts 为空本身不构成错误。)",
    "feedback": "具体指明错误点。若无实质错误，返回 '通过审计，无需修改'。禁止输出'请确认是否没有新事实'这类非错误性质建议。",
    "contradiction_flags": "可选字符串列表，列举与 research_facts 直接冲突的条目 id 或描述；无冲突则返回 []"
}}
"""
_fact_harvester_evaluator_llm = get_llm(temperature=0.1, top_p=0.5)
async def fact_harvester_evaluator(state: ConsolidatorState) -> dict:
    system_prompt = SystemMessage(_FACT_HARVESTER_EVALUATOR_PROMPT.format(
        character_name=state["character_profile"]["name"],
        user_name=state["user_name"],
    ))

    retry = state.get("fact_harvester_retry", 0) + 1
    msg = {
        "retry": f"{retry}/{MAX_FACT_HARVESTER_RETRY}",
        "new_facts": state["new_facts"],
        "future_promises": state["future_promises"],

        "decontexualized_input": state["decontexualized_input"],
        "research_facts": state["research_facts"],
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
