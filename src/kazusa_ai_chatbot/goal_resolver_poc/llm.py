"""LLM role calls for the goal resolver POC."""

from __future__ import annotations

import asyncio
import ast
import json
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from openai import BadRequestError

from kazusa_ai_chatbot.config import (
    RAG_PLANNER_LLM_API_KEY,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MODEL,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.goal_resolver_poc.models import (
    bounded_text,
    normalize_case_evaluation,
    normalize_patch_output,
    normalize_planner_output,
    normalize_verifier_output,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output


LLM_CALL_TIMEOUT_SECONDS = 90


_PLANNER_PROMPT = '''\
你是一个通用目标解析器的 planner。你的任务不是扮演角色回答用户，而是选择下一步解析动作。

# 核心原则
1. 只根据用户输入、当前需求状态、工具历史和 verifier 反馈推进目标。
2. 不要为了某个测试样例写特殊规则；你面对的是任意真实用户任务。
3. 每轮只选择一个最有价值的动作，并把动作绑定到一个 open requirement。
4. 如果缺少用户拥有的信息，而且继续调用工具无法可靠补齐，选择 ask_human。
5. 如果用户明确允许你使用假设或说明限制，不要因为这个假设去 ask_human；写清楚假设并继续推进。
6. 如果任务需要写入、发送、调度、修改生产代码或其他副作用，选择 prepare_action 或 ask_human，不要执行副作用。
7. 如果已有证据足够满足目标，或公开/内部证据已经合理尝试但仍无法确认，选择 final_answer，并要求最终答复说明证据限制。
8. 不要重复相同工具和相同 query；如果需要重试，必须改变 query 或说明新目的。
9. 如果已有观察已经包含失败输出、源内容、日志内容或动作候选，不要重复读取同一材料；选择下一个能推进状态的动作。
10. 如果多轮不同查询都无法确认某项公开或内部事实，不要无限继续搜索；选择 final_answer 并说明已尝试范围和来源限制。
11. 如果用户要求系统检查自身可用记录，但多轮直接和宽泛检索都没有证据，不要要求用户补搜索词；选择 final_answer，状态写 unknown/证据不足，并说明检索范围。
12. final_answer 不是变相追问。除非 terminal mode 是 ask_human，否则 final_answer 必须尽力完成用户目标；如果用户允许假设或说明限制，应在假设下继续产出结论。
13. 不要把全部循环预算耗在一个可声明假设的前提上。若同一前提经过两次不同检索仍无法确认，且用户允许假设/限制，下一步应转向其他 open requirement，并把该前提作为假设或来源限制带入后续动作。
14. 如果同一内部或公开事实已经经过三轮以上不同检索仍只有缺少上下文、无已确认事实或候选不足，下一步不要继续换近义词检索；选择 final_answer，让最终答复明确证据不足和已检查范围。
15. 如果 verifier feedback 明确指出 stuck in loop、repeated、exhausted、should terminate、final_answer 或同义内容，下一步必须选择 final_answer，除非缺口属于真正需要用户回答或批准的事项。
16. 对本地餐厅、营业时间或临时可去性任务，如果已有公开目录证据包含店名、地址、评分、评论数、当天营业时间和来源 URL，就选择 final_answer 给出可执行建议和时效限制；不要为了无法确认实时座位或 walk-in 保证而无限搜索，除非用户明确要求保证空位或预约状态。
17. 不要把 planner 自己扩张出来的更强要求当作阻塞项。用户只要求“临时可决定”“可能营业”“评价别太差”时，不能升级为必须证明 guaranteed walk-in、实时空位或预约状态；这些只能作为最终答复里的限制和建议。
18. 对软件项目正式发布日期任务，如果观察中有 GitHub Releases API 的非 draft、非 prerelease 发布行，下一步选择 final_answer；最终答复要给出 tag、published_at 日期、URL，并说明已排除 draft/prerelease。
19. 对 self_goal_generate 的观察，如果它已经包含 selected_goal、selection_reason、completed_result 或 verification_result，下一步应选择 final_answer 汇报处理状态；不要再选择 workspace_command，除非用户或观察明确要求运行仓库命令且该命令在批准的 fixture 范围内。
20. 如果已经通过 ask_human 提出最小关键问题，并且正在等待用户回答，不要重复 ask_human；选择 ask_human 终态并让 finalizer 说明等待的最小问题。
21. 对本地餐厅任务，如果只剩“是否有实时空位、是否保证 walk-in、是否有当天临时关店公告”这类无法从公开目录稳定确认的条件，且用户没有明确要求保证，选择 final_answer，并把这些写成出发前确认的 caveat。
22. 对代码修复任务，如果已经执行 workspace_patch，下一步必须运行 workspace_command 重新验证，除非已经有 patch 之后的 workspace_command 成功输出。
23. 禁止为用户没有明确写出的餐厅保证创建 requirement。只有当 user_input 本身包含“预约”“空位”“walk-in”“无需预约”“保证”“reservation”“booking”等词时，才可以把实时座位、walk-in 或预约政策作为阻塞 requirement。
24. 对本地硬件推荐任务，如果观察中已有 hardware_catalog_evidence 或 public_catalog_fallback，且包含本地可购买的合适 GPU 或整机，不要要求更便宜替代品，除非用户明确写了预算、便宜、性价比、alternative、cheapest、budget 等约束。
25. 对硬件整机或工作站证据，如果产品标题或类型已经包含 GPU、CPU、内存和存储线索，可以进入 final_answer 形成一套完整建议；未逐项证明的 PSU、机箱、散热和现货变化应作为限制说明，而不是无限继续搜索。
26. 如果用户要求一套电脑、硬件 setup、硬件配置、build 或采购清单，不要只用 GPU 证据结束；必须继续取得整机/workstation 证据，或至少取得 CPU、内存、存储、电源/机箱这些配套项的证据。只有用户明确只问显卡时才可只给 GPU。
27. 如果用户已经允许量化版本、运行方式假设或限制说明，不要再把“是否接受量化”当作 HIL 问题；按量化推理给出建议，并把 full precision 写成不满足当前本地单卡约束的限制。
28. 对本地日志、报告、artifact 或 RCA 任务，优先选择 local_artifact_inspect。workspace_command 只用于用户明确给出验证命令或需要在批准沙盒中运行命令的代码修复任务。

# 可用工具
- rag_research: 检索内部记忆、人物关系、聊天证据、recall，必要时也能走现有检索证据。
- web_research: 检索公开互联网信息，适合当前事实、网页内容、公开资料和可购买性证据。
- workspace_inspect: 只读检查工作区或隔离沙盒文件。
- workspace_command: 在隔离沙盒或只读验证场景中运行允许的验证命令。
- workspace_patch: 只在隔离沙盒中根据失败证据生成并应用补丁。
- local_artifact_inspect: 读取本地日志、报告或运行 artifacts。
- self_goal_generate: 为“自己想一个目标”生成有限、可验证、不含对外发送的目标。
- prepare_action: 准备需要人类批准的动作候选，不执行。
- ask_human: 生成最小人类问题并暂停。
- final_answer: 进入最终回答生成。

# requirements 输出规则
- requirements 是你对用户目标的当前分解。它可以保留已有 requirement_id，也可以补充更细的 requirement。
- 每个 requirement 必须是可被证据、HIL 或 approval 状态验证的要求。
- 不要把工具名写成 requirement；写用户目标中必须满足的条件。
- 输入里的 requirement_state 是当前状态，只供判断下一步使用；输出 requirements 时禁止复制其中的 status、blocking_reason、satisfied_by_observation_ids 或 last_verifier_note。

# 输出格式
只返回 JSON 对象。第一个字符必须是 `{`，禁止使用 ```json 或任何 markdown 代码围栏：
{
  "goal_frame": "一句话描述当前目标",
  "requirements": [
    {
      "requirement_id": "req-001",
      "description": "必须满足的条件",
      "required_evidence_type": "需要的证据类型"
    }
  ],
  "open_requirements": ["仍需满足的关键条件"],
  "next_action": {
    "tool": "rag_research | web_research | workspace_inspect | workspace_command | workspace_patch | local_artifact_inspect | self_goal_generate | prepare_action | ask_human | final_answer",
    "target_requirement_id": "req-001",
    "query": "给工具的具体输入；ask_human 时写要问用户的问题；final_answer 时写最终回答应覆盖什么",
    "reason": "为什么此刻选择这个动作"
  }
}
'''
_planner_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=RAG_PLANNER_LLM_MODEL,
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
)
_planner_fallback_llm = get_llm(
    temperature=0.1,
    top_p=0.9,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


def _bad_request_response_content(exc: BadRequestError) -> str:
    """Recover JSON content embedded in a local endpoint parse error."""

    error_text = str(exc)
    delimiter = " - "
    if delimiter in error_text:
        _, payload_text = error_text.split(delimiter, maxsplit=1)
        try:
            payload = ast.literal_eval(payload_text)
        except (SyntaxError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            payload_error = payload.get("error")
            if isinstance(payload_error, str):
                error_text = payload_error

    marker = "Failed to parse input at pos 0:"
    marker_index = error_text.find(marker)
    if marker_index < 0:
        return ""
    content = error_text[marker_index + len(marker):].strip()
    if not content.startswith("{"):
        return ""
    try:
        parse_llm_json_output(content)
    except (json.JSONDecodeError, ValueError, TypeError):
        return ""
    return content


async def _ainvoke_with_bad_request_fallback(
    primary_llm: Any,
    fallback_llm: Any,
    messages: list[SystemMessage | HumanMessage],
) -> Any:
    """Invoke an LLM and retry alternate transport on bounded failures."""

    last_error: Exception | None = None
    for llm in (primary_llm, fallback_llm, primary_llm):
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=LLM_CALL_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, BadRequestError) as exc:
            if (
                isinstance(exc, BadRequestError)
                and "Failed to parse input" not in str(exc)
            ):
                raise
            if isinstance(exc, BadRequestError):
                recovered_content = _bad_request_response_content(exc)
                if recovered_content:
                    response = SimpleNamespace(content=recovered_content)
                    return response
            last_error = exc
            continue
        return response

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM invocation failed without an exception")


_PLANNER_REPAIR_PROMPT = '''\
你是 planner JSON 修复器。输入包含一个 schema_error 和当前 resolver state。
你只负责返回一个符合 planner schema 的 JSON 对象，不要解释。

# 修复规则
1. requirements 只能包含 requirement_id、description、required_evidence_type。
2. requirements 里不要输出 status、blocking_reason、satisfied_by_observation_ids、last_verifier_note。
3. next_action.tool 必须从允许工具中选择一个，不能空。
4. 如果还有 open requirement，选择能推进该 requirement 的工具。
5. 如果 verifier feedback 说明只剩最终综合、总结、报告或回答，选择 final_answer。
6. 如果缺少用户拥有且工具无法补齐的信息，选择 ask_human。
7. 如果需要副作用确认，选择 prepare_action。

# 允许工具
rag_research, web_research, workspace_inspect, workspace_command,
workspace_patch, local_artifact_inspect, self_goal_generate, prepare_action,
ask_human, final_answer

# 输出格式
只返回 JSON 对象。第一个字符必须是 `{`，禁止 markdown 代码围栏：
{
  "goal_frame": "当前目标",
  "requirements": [
    {
      "requirement_id": "req-001",
      "description": "必须满足的条件",
      "required_evidence_type": "证据类型"
    }
  ],
  "open_requirements": ["仍需满足的条件"],
  "next_action": {
    "tool": "允许工具之一",
    "target_requirement_id": "req-001",
    "query": "工具输入或最终回答要覆盖的内容",
    "reason": "选择原因"
  }
}
'''


async def call_planner(state_view: dict[str, Any]) -> dict[str, Any]:
    """Call the generic planner role.

    Args:
        state_view: Bounded resolver state visible to the planner.

    Returns:
        Normalized planner output with a next action and requirement rows.
    """

    payload = json.dumps(state_view, ensure_ascii=False)
    messages = [
        SystemMessage(content=_PLANNER_PROMPT),
        HumanMessage(content=payload),
    ]
    response = await _ainvoke_with_bad_request_fallback(
        _planner_llm,
        _planner_fallback_llm,
        messages,
    )
    raw = parse_llm_json_output(str(response.content))
    try:
        normalized = normalize_planner_output(raw)
    except ValueError as exc:
        repair_payload = {
            "schema_error": str(exc),
            "repair_instruction": (
                "Return a valid planner JSON object with a non-empty "
                "next_action.tool selected from the allowed tool roster."
            ),
            "state": state_view,
        }
        repair_messages = [
            SystemMessage(content=_PLANNER_REPAIR_PROMPT),
            HumanMessage(
                content=json.dumps(repair_payload, ensure_ascii=False)
            ),
        ]
        last_error: Exception = exc
        for repair_llm in (_planner_fallback_llm, _planner_llm):
            try:
                response = await asyncio.wait_for(
                    repair_llm.ainvoke(repair_messages),
                    timeout=LLM_CALL_TIMEOUT_SECONDS,
                )
            except BadRequestError as repair_exc:
                if "Failed to parse input" not in str(repair_exc):
                    raise
                last_error = repair_exc
                continue
            raw = parse_llm_json_output(str(response.content))
            try:
                normalized = normalize_planner_output(raw)
            except ValueError as repair_exc:
                last_error = repair_exc
                continue
            break
        else:
            raise last_error
    normalized["raw_output"] = str(response.content)
    return normalized


_VERIFIER_PROMPT = '''\
你是一个通用目标解析器的 verifier。你不生成最终答复；你判断当前状态是否已经足够接近用户目标。

# 判断标准
1. 对照用户输入和 requirements，不要只看工具是否返回了内容。
2. 每个 requirement 必须更新为 open、satisfied、blocked_human、blocked_approval 或 unresolved。
3. satisfied 必须引用支持它的 observation id。
4. blocked_human 只能用于缺少用户拥有且工具无法补齐的信息。
5. blocked_approval 只能用于需要用户确认的副作用或动作候选。
6. unresolved 只能用于已经合理尝试但仍无法确认的要求，最终答复必须承认证据限制。
7. 如果继续检索仍有价值，decision 必须是 continue 并给出下一轮方向。
8. 如果用户允许使用假设或说明限制，不要把缺少假设确认判为 blocked_human。
9. 如果已有观察足以支持分析、修复、验证或动作候选，必须把对应 requirement 标为 satisfied、blocked_human 或 blocked_approval，不要保持 open。
10. 如果公开或内部事实经过多轮有差异的检索仍无法确认，必须把 requirement 标为 unresolved，并允许 final_answer 说明无法确认；不要为了来源缺失请求用户补搜索结果。
11. 当用户要求系统自己检查可用记录但没有给出具体对象时，缺少对象名不自动构成 blocked_human；如果宽泛检索无证据，最终状态应是 unknown/证据不足。
12. 如果用户要求直接产出建议、结论、修复或根因分析，且允许使用假设或说明限制，不能把 final_answer 写成让用户确认前提；这类状态仍有 open requirement。
13. 若同一可假设前提已经多轮检索失败，但用户允许假设/限制，应把该前提标为 unresolved 并要求 planner 转向其余 requirement，而不是继续重复查询。
14. 如果最新观察是 ask_human，必须把被该问题阻塞的 requirement 标为 blocked_human。
15. 如果最新观察是 prepare_action，必须把需要用户确认的副作用或动作 requirement 标为 blocked_approval。
16. 如果某个 requirement 的内容是“给出建议、总结、回答、报告、清单或结论”，且所需证据已经存在，必须把它标为 satisfied；最终文字由 finalizer 生成，不要只因为 finalizer 尚未写出正文而保持 open。
17. 对硬件配置类任务，如果观察中已经有完整 PC、workstation、desktop 或 gaming PC 产品，且产品名或字段包含所需 GPU、CPU、内存和存储，它可以满足“完整配置/整机方案”要求；最终答复应说明这是整机方案，未逐项列出的 PSU、机箱和散热以该整机配置为准。
18. 如果最近至少三次不同检索都显示“没有找到已确认事实”“缺少必要上下文”“候选不足以确认”或同义结果，不要继续要求 planner 做元检查；把相关事实要求标为 unresolved，把“如证据不足则说明不足”的要求标为 satisfied 或 unresolved，并允许 final_answer。
19. 对本地餐厅、营业时间或临时可去性任务，若观察中已有公开目录证据列出店名、评分、评论数、地址、当天营业时间和 URL，应把“可能营业”和“评价不差”要求标为 satisfied；若没有实时空位或 walk-in 证据，只把它作为 final_answer 的限制说明，不要保持 open，除非用户明确要求保证空位。
20. 如果 planner 生成的 requirement 比用户原话更强，例如把“临时去/直接决定”升级成“必须证明无需预约、保证 walk-in 或实时空位”，而用户没有明确要求保证，则不要让这个扩张要求阻塞终态；标为 satisfied 或 unresolved，并要求 final_answer 明确“需要出发前再电话/官网确认”。
21. 对软件项目正式发布日期任务，如果观察中已有 GitHub Releases API 证据，且 release 行显示 draft=false、prerelease=false、published_at 和 html_url，应把发布日期要求标为 satisfied，并允许 final_answer。不要继续依赖搜索摘要。
22. 对 self_goal_generate 的观察，如果 payload 已包含 selected_goal、selection_reason、completed_result 或 verification_result，并且该目标不需要外部发送或生产副作用，应把“生成目标、说明理由、执行/验证、汇报状态”相关要求标为 satisfied。不要要求 workspace_command 来证明一个可由观察本身验证的内部目标。
23. 对 HIL 场景，如果最新观察是 ask_human 且问题覆盖了制定方案所需的最小关键缺口，必须把被等待用户回答的 requirement 标为 blocked_human；用户未要求立即确认的可选偏好、禁忌或细节不能保持 open，应标为 blocked_human 或 satisfied，并在 final_answer 说明可在用户回答后继续细化。
24. 对本地餐厅任务，公开目录是在本轮工具调用中读取的；若没有证据显示当天闭店或节假日影响，不要要求额外证明“没有临时闭店公告”。实时空位和 walk-in 保证只能作为 caveat，不能阻塞 final_answer，除非用户明确要求保证空位。
25. 对代码修复任务，“重新运行验证命令并通过”只能由 workspace_command 观察满足，且必须发生在 workspace_patch 之后，returncode 必须为 0。workspace_patch 本身不能满足 rerun verification requirement。
26. 若 user_input 没有明确包含“预约”“空位”“walk-in”“无需预约”“保证”“reservation”“booking”等词，但 planner 生成了相关 requirement，该 requirement 是过度扩张；必须标为 satisfied 或 unresolved，并且 decision 必须允许 final_answer，不能 continue。
27. 对本地硬件推荐任务，若 hardware_catalog_evidence 中有 available_graphics_products 或 available_ready_systems，且其中有满足用户核心硬件目标的候选，应把“本地可购买性”标为 satisfied。不要因为没有搜索更便宜、更多品牌或上一代卡而保持 open，除非用户明确要求预算、便宜、性价比或替代方案。
28. 对硬件完整配置 requirement，available_ready_systems 可以满足整机方案；available_component_products 可以补充 CPU、内存、SSD、PSU 等零件线索。最终答复可把未确认的细节写成 caveat，但不能把 Python/LLM 可根据证据综合的最终建议本身当成 open。
29. 对本地硬件推荐任务，如果最近多轮反馈都在要求“继续找兼容组件”，而 tool_history 已经有 available_ready_systems 或 CPU、内存、SSD、PSU 等 available_component_products，应停止继续搜索，把完整配置 requirement 标为 satisfied 或 unresolved caveat，并允许 final_answer。
30. 如果本地日志、报告、artifact 或 RCA 任务已经通过 local_artifact_inspect 读到文件列表和内容，必须围绕这些文件判断根因；不要再要求用户移动文件或粘贴内容。
31. 对硬件 setup/build/配置任务，如果观察中只有 GPU 证据，没有整机/workstation 或 CPU、内存、存储、电源/机箱配套证据，不得把完整硬件配置 requirement 标为 satisfied；必须继续检索配套项或把最终答复限制为“不完整，仅显卡已确认”。
32. 如果用户已经允许量化版本、运行方式假设或限制说明，不要把“是否接受量化”标为 blocked_human；应把量化作为 satisfied 假设继续，full precision 只能作为限制说明。
33. 对 27B 级模型硬件答复，如果 final_answer 把 24GB 或 32GB 级消费显卡写成 FP16、全精度、full precision 或接近 FP16 的稳妥方案，且不是明确否定这种能力，必须判为未满足并要求修正；这类显卡只能作为量化推理建议，除非证据明确支持更高精度。

# 输出格式
只返回 JSON 对象。第一个字符必须是 `{`，禁止使用 ```json 或任何 markdown 代码围栏：
{
  "resolved": true,
  "decision": "continue | final_answer | ask_human | prepare_action",
  "confidence": 0.0,
  "requirement_updates": [
    {
      "requirement_id": "req-001",
      "status": "open | satisfied | blocked_human | blocked_approval | unresolved",
      "blocking_reason": "如果被阻塞或未解决，说明原因",
      "satisfied_by_observation_ids": ["obs-001"],
      "last_verifier_note": "判断说明"
    }
  ],
  "remaining_requirements": ["仍未满足的条件"],
  "feedback": "给 planner 或 finalizer 的具体反馈",
  "minimal_human_question": "如果 decision=ask_human，在这里写最小问题，否则空字符串"
}
'''
_verifier_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def call_verifier(state_view: dict[str, Any]) -> dict[str, Any]:
    """Call the generic verifier role."""

    payload = json.dumps(state_view, ensure_ascii=False)
    response = await _ainvoke_with_bad_request_fallback(
        _verifier_llm,
        _planner_llm,
        [
            SystemMessage(content=_VERIFIER_PROMPT),
            HumanMessage(content=payload),
        ],
    )
    raw = parse_llm_json_output(str(response.content))
    normalized = normalize_verifier_output(raw)
    normalized["raw_output"] = str(response.content)
    return normalized


_FINALIZER_PROMPT = '''\
你是 goal resolver 的最终答复生成器。根据用户输入、目标状态、工具观察和 verifier 反馈写一段自然语言结果。

# 规则
1. 不要声称工具没有证明的事实。
2. 如果最终状态是 needs_human，只写清楚缺什么、为什么必须问、最小问题是什么。
3. 如果最终状态是 pending_approval，只写准备做什么、为什么、影响是什么、等待用户确认。
4. 如果证据不足但已经合理停止，明确说明证据不足和已检查范围。
5. 对本地日志或代码类任务，引用文件名、命令或关键字段。
6. 对公开网页任务，保留来源名称、URL 或检索证据限制。
7. final 状态不能伪装成 HIL。除非状态是 needs_human，否则不要把主要内容写成“请用户确认后我再做”；应在已有证据和明确假设下给出当前可执行结论。
8. 不要输出 markdown 表格；用短段落或项目符号表达即可。
9. 对 LLM 硬件部署任务，必须区分量化推理、8-bit、FP16/full precision 等运行假设；不要声称某个显存容量支持 full precision，除非工具证据明确支持。27B 级模型的 FP16/full precision 权重本身约 54GB 以上，单张 24GB 或 32GB 消费级显卡不能作为 FP16/full precision 方案。24GB 或 32GB 级消费显卡只能作为量化推理建议；8-bit 也必须写成可能吃满显存、需要实测和较小上下文，而不是保证。
10. 对软件 release 日期任务，优先使用 GitHub Releases API 的 published_at；如果 release 行显示 draft=false 且 prerelease=false，可说明这是正式发布。给出 tag/name、发布日期、URL，并说明没有把搜索摘要、预告或 prerelease 当作结论。
11. 对本地硬件推荐任务，优先把已观察到的本地可购买 GPU 或整机写成一套可执行配置。只有用户明确要求预算或替代方案时才把缺少更便宜选择当作主要缺口；否则用“可选替代/未确认”表达来源限制。
12. 对硬件 setup/build/配置任务，最终答复必须覆盖 GPU、CPU、内存、存储、电源/机箱，或明确给出一个已观察到的整机/workstation 产品。没有证据的配套项只能写成“需向零售商确认”，不能写成已确认配置。
13. 如果用户已经允许量化版本、运行方式假设或限制说明，最终答复不要追问是否接受量化；直接按量化推理给出可执行方案，并把 full precision 写成当前约束下不建议或不可满足。
14. 引用产品名、型号、价格、URL、发布日期或版本号时，必须从工具观察中逐字复制；不要凭记忆改写型号数字。如果不确定，就写“证据中看到的型号为 ...”，并保留原始字符串。
15. 输出前自检：如果你的草稿把 24GB 或 32GB 级消费显卡写成 FP16、全精度、full precision 或接近 FP16 的方案，并且语义不是“不能/不建议/不足以”，必须改写为量化推理限制说明。

输出自然语言文本，不要 JSON，不要 markdown 表格。
'''
_finalizer_llm = get_llm(
    temperature=0.1,
    top_p=0.9,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def call_finalizer(state_view: dict[str, Any]) -> str:
    """Call the final answer generator."""

    payload = json.dumps(state_view, ensure_ascii=False)
    response = await asyncio.wait_for(
        _finalizer_llm.ainvoke(
            [
                SystemMessage(content=_FINALIZER_PROMPT),
                HumanMessage(content=payload),
            ]
        ),
        timeout=LLM_CALL_TIMEOUT_SECONDS,
    )
    final_answer = str(response.content).strip()
    return final_answer


_CASE_EVALUATOR_PROMPT = '''\
你是 goal resolver POC 的独立 LLM 评审员。你评估的是“这个运行是否解决了用户任务”，不是评估文字是否好看。

# 输入会包含
- 用户的自然语言输入
- 本 case 的 resolver contract
- 允许的 terminal modes
- 完整 resolver trace
- 最终答复

# 评审原则
1. 如果 terminal mode 不在允许列表中，失败。
2. 如果用户目标需要证据，检查 trace 中是否有足够工具观察支撑最终答复。
3. 如果最终答复选择证据不足，必须有合理的检索或缺口说明。
4. 如果是 HIL，有效条件是缺失信息确实属于用户拥有，且问题最小。
5. 如果是 pending approval，有效条件是没有执行副作用，并且说明了动作、理由和影响。
6. 不要因为 Python artifact 形状完整就判通过；必须语义判断。
7. 如果 terminal mode 是 final，但最终答复主要是在追问用户或承诺之后再做，而不是完成用户要求，则应判 fail，除非 case contract 明确允许证据不足作为最终结论。
8. 如果用户明示可以使用假设或说明限制，因可声明的假设而停止追问应判 fail。
9. 对硬件推荐任务，若 trace 中有技术运行假设、公开本地零售证据支持至少一个可购买 GPU 或整机，并且最终答复给出完整可执行建议和限制说明，不要因为缺少更便宜替代品而判 fail，除非用户明确要求预算或替代方案。
10. 对 27B 级模型硬件任务，如果最终答复把 24GB 或 32GB 级消费显卡写成可以稳妥运行 FP16、全精度、full precision 或接近 FP16，且不是明确否定这种能力，必须判 fail；量化推理建议与 full precision 建议必须分开。
11. 如果最终答复引用具体产品名、型号、价格、URL、版本号或发布日期，却与 trace 中的工具观察不一致，且该字段支撑用户关键目标，必须判 fail 或列为 missing；不能用大意正确掩盖型号数字错误。
12. 如果用户要求一套硬件 setup/build/配置，而最终答复只有 GPU、本地购买渠道和泛泛的内存/电源建议，没有整机/workstation 证据，也没有 CPU、内存、存储、电源/机箱的配套证据，应判 fail 或列为 missing；只有用户明确只问显卡时例外。
13. 如果用户已经允许量化版本、运行方式假设或限制说明，而最终答复主要是在追问用户是否接受量化，应判 fail；正确行为是按量化假设给出当前可执行建议并说明 full precision 限制。
14. score 必须是 0 到 100 的整数；pass 或 needs_human_valid 通常应为 80-100，fail 应低于 80。

# 输出格式
只返回 JSON 对象。第一个字符必须是 `{`，禁止使用 ```json 或任何 markdown 代码围栏：
{
  "status": "pass | fail | needs_human_valid",
  "score": 0,
  "reason": "评审理由",
  "missing": ["缺失项"],
  "loop_quality": "对循环质量的简短评价",
  "tool_use_quality": "对工具使用质量的简短评价"
}
'''
_case_evaluator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def call_case_evaluator(state_view: dict[str, Any]) -> dict[str, Any]:
    """Call the independent final case evaluator."""

    payload = json.dumps(state_view, ensure_ascii=False)
    response = await _ainvoke_with_bad_request_fallback(
        _case_evaluator_llm,
        _planner_llm,
        [
            SystemMessage(content=_CASE_EVALUATOR_PROMPT),
            HumanMessage(content=payload),
        ],
    )
    raw = parse_llm_json_output(str(response.content))
    normalized = normalize_case_evaluation(raw)
    normalized["raw_output"] = str(response.content)
    return normalized


_SELF_GOAL_PROMPT = '''\
你为一个角色内部 resolver 生成并完成候选小目标。目标必须有限、可验证、不能需要对外发送消息，不能需要生产副作用，也不能依赖未批准的 workspace_command。

选择目标时优先选择能在当前输入、当前状态、已有工具观察或纯文本推理中完成验证的目标。不要选择需要运行仓库命令、修改文件、访问外部账号或等待用户回复的目标。
你只能使用输入 JSON 中实际提供的字段。禁止声称已经读取文件系统、扫描工作区、访问数据库、运行命令、联网检索或查看未出现在输入里的材料。若没有文件列表，就不能选择“检查文件结构”类目标。

只返回 JSON 对象。第一个字符必须是 `{`，禁止使用 ```json 或任何 markdown 代码围栏：
{
  "candidates": [
    {"goal": "目标", "why": "为什么值得处理", "verification": "如何验证完成"}
  ],
  "selected_goal": "最终选择的目标",
  "selection_reason": "选择理由",
  "completed_result": "你已经处理完成的具体结果",
  "verification_result": "为什么这个结果可验证且已完成"
}
'''
_self_goal_llm = get_llm(
    temperature=0.2,
    top_p=0.9,
    model=RAG_PLANNER_LLM_MODEL,
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
)


async def call_self_goal_generator(state_view: dict[str, Any]) -> dict[str, Any]:
    """Generate bounded self-goal candidates."""

    payload = json.dumps(state_view, ensure_ascii=False)
    response = await asyncio.wait_for(
        _self_goal_llm.ainvoke(
            [
                SystemMessage(content=_SELF_GOAL_PROMPT),
                HumanMessage(content=payload),
            ]
        ),
        timeout=LLM_CALL_TIMEOUT_SECONDS,
    )
    raw = parse_llm_json_output(str(response.content))
    if not isinstance(raw.get("candidates"), list):
        raw["candidates"] = []
    for key in ["selected_goal", "selection_reason", "completed_result"]:
        value = raw.get(key)
        if isinstance(value, str):
            continue
        if value is not None:
            raw[key] = json.dumps(value, ensure_ascii=False)
        else:
            raw[key] = ""
    value = raw.get("verification_result")
    if not isinstance(value, str) and value is not None:
        raw["verification_result"] = json.dumps(value, ensure_ascii=False)
    elif value is None:
        raw["verification_result"] = ""
    raw["raw_output"] = str(response.content)
    return raw


_SANDBOX_PATCH_PROMPT = '''\
你是隔离沙盒里的代码修复助手。根据文件内容和失败输出，返回要写入的完整文件内容。

# 规则
1. 只修复失败所需的最小问题。
2. 不要改动无关文件。
3. 只返回 JSON。第一个字符必须是 `{`，禁止使用 ```json 或任何 markdown 代码围栏。

# 输出格式
{
  "file_path": "相对沙盒根目录的文件路径",
  "new_content": "完整文件内容",
  "reason": "为什么这样修"
}
'''
_sandbox_patch_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def call_sandbox_patcher(state_view: dict[str, Any]) -> dict[str, Any]:
    """Ask the LLM for a sandbox-only patch."""

    payload = json.dumps(state_view, ensure_ascii=False)
    response = await asyncio.wait_for(
        _sandbox_patch_llm.ainvoke(
            [
                SystemMessage(content=_SANDBOX_PATCH_PROMPT),
                HumanMessage(content=payload),
            ]
        ),
        timeout=LLM_CALL_TIMEOUT_SECONDS,
    )
    raw = parse_llm_json_output(str(response.content))
    raw["raw_output"] = str(response.content)
    normalized = normalize_patch_output(raw)
    return normalized
