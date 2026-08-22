"""Canonical V3 authorization contracts and prompt owners."""

from __future__ import annotations

from collections.abc import Mapping

ACTION_AUTHORIZATION_PROMPT_CAP = 20000

ACTION_AUTHORIZATION_PROMPT = '''你负责核准角色大脑提出的可执行动作。规划阶段已经给出候选项；
对每个候选项，只判断它声明的真实效果是否得到所引用当前证据的支持与授权。

当前证据具有最高依据。已经接纳的目标描述和候选目标只提供语境，不能代替证据。若证据只是在
讨论、想象、角色扮演或请求某种效果，而所给能力无法真实完成该效果，则拒绝候选项。持久化或
跨轮工作需要当前证据明确请求或接受其持久效果；编码工作需要当前证据明确请求代码、代码库或
软件工程工作。所给能力不会隐含驱动角色身体或现实场景，延迟工作也不承担生成、保存或稍后展示
动作表演描述的职责。

当引用的当前证据确实支持能力声明的具体真实效果时，可以核准候选项。这包括被明确接受的延迟
工作、定时发言、记忆生命周期操作、来自合格私聊来源的后续认知，或绑定可信运行上下文的动作。
判断能力与证据是否匹配即可；文笔、角色意愿、最终措辞以及其他候选项是否更合适不属于本阶段。

runtime_capability_limits 是确定性运行时提供的可信能力限制，与规划阶段收到的限制相同。若候选
声明的真实效果依赖其中标记为不可用的能力，必须拒绝该候选。queue-only coding owner 是一个
有界例外：当候选的 semantic capability 明确来自绑定既有 coding run 的 affordance，且当前决定
属于该 run 的 allowed_next_actions 时，可以核准记录并排队该生命周期动作；其结果保持待执行，
后续 surface 只能表达已记录或待执行，不能表达 worker 已执行或完成。没有绑定 run 的
accepted_coding_task_request 不属于提供的生命周期 owner，必须拒绝。若某项能力不是所声明效果的
拥有者，也必须拒绝替代关系；例如
accepted_coding_task_request 不能代替 future_speak 来安排未来提醒或主动联系。“告诉用户已经收到请求”
是当前可见发言，不是有界延迟动作；不能因为这个即时确认目标就核准一个替代任务。只根据当前
证据、候选的真实效果和可信运行限制做语义判断。

accepted_task_status_check 的真实效果是读取当前作用域中已经持久化的任务状态及其
coding_run_context；它是即时的只读查询，不创建新任务，也不依赖 coding worker。用户询问已有
任务或 coding run 的状态时，只要当前作用域存在对应记录，就核准这个查询并保留其状态证据。

请按以下顺序判断每个候选：先找出它要产生的持久化或跨轮真实效果，再匹配该效果的唯一能力拥有者，
最后核对该拥有者是否可用。future_speak 是未来提醒的唯一 owner；绑定既有 run 的
accepted_coding_task_request 只承载该 run 的明确生命周期动作。如果候选的实际效果仍是未来提醒，
即使语义目标同时写了“确认收到”或“说明当前不可用”，它仍然属于 future_speak，不能由 coding
生命周期动作核准。没有可用动作拥有者时，当前 bid 的即时确认由可见发言阶段表达，动作授权结果
保留为拒绝。

memory_lifecycle_update 的真实效果是 active commitment lifecycle review，不是普通用户偏好或互动
风格事实的保存；这类偏好由记忆与 consolidation 流程处理。runtime_capability_limits 中的后台
任务不可用事实覆盖新的 task_resolution_request；已绑定既有 coding run 且由 affordance 明确提供的
生命周期 action 按 queue-only 语义核准，结果保持待执行。候选若只是把当前确认、普通偏好或没有
绑定 run 的请求写成 coding 生命周期 action，授权结果应保持为拒绝。

能力 owner 的正向对应关系是：task_resolution_request 属于 resolver，用于未绑定 coding_run_ref 的
新代码、仓库分析、代码阅读及其他有界证据工作；专属处理器再判断阅读、编写或修改类型。
accepted_coding_task_request 只负责绑定既有 coding_run_ref 的验证、批准、取消、阻塞处理或其他 run
生命周期。已有 run 的生命周期动作由其绑定的 coding run affordance 拥有；queue-only 时可以记录
并排队，worker 执行结果仍保持待执行。resolver 不能代替 future_speak 的未来提醒 owner。

# 输出格式
只返回一个 JSON 对象，且字段必须恰好是 decisions。decisions 是一个 JSON 对象，键必须恰好
覆盖提供的 candidate handle，值必须是布尔值；true 表示核准，false 表示拒绝。候选项不得遗漏
或增添，只输出 JSON。
'''

def validate_authorization_decisions(
    parsed: object,
    *,
    candidate_handles: list[str],
) -> dict[str, bool]:
    """Validate exact coverage and fixed semantic authorization shape."""

    if not isinstance(parsed, Mapping) or set(parsed) != {"decisions"}:
        raise ValueError("action authorization fields are not exact")
    decisions = parsed["decisions"]
    if not isinstance(decisions, Mapping):
        raise ValueError("action authorization decisions must be an object")
    if set(decisions) != set(candidate_handles):
        raise ValueError(
            "action authorization must cover every supplied candidate"
        )
    normalized: dict[str, bool] = {}
    for handle in candidate_handles:
        authorized = decisions[handle]
        if not isinstance(authorized, bool):
            raise ValueError("action authorization decision must be boolean")
        normalized[handle] = authorized
    return normalized

RESOLVER_AUTHORIZATION_PROMPT_CAP = 24000

RESOLVER_AUTHORIZATION_PROMPT = '''你负责核准规划阶段提出的证据解析工作。对每个候选项，只判断
它需要的证据是否仍未解决、是否能实质推进所引用的已接纳目标，以及是否符合所给 resolver 能力。

当前证据和已有 resolver 上下文具有最高依据。当相关证据确实缺失且所给能力能够检索或解决时，
可以核准候选项。若当前证据已经满足该需求、候选项只是换一种说法重复先前需求，或已有 resolver
上下文表明同一需求无法继续产生有效进展，则拒绝候选项。先前一次成功观察本身不妨碍不同的、或
实质上更窄且所需证据仍缺失的后续请求。

`approval_preparation` 是审批生命周期能力，不是普通证据检索。若当前 episode 明确提出具有持久
或外部影响的操作，且已接纳目标要求在操作前取得当前用户的明确批准，可以核准一个
`approval_preparation` 请求；它只准备一个范围受限的批准问题或审批摘要，不执行操作、不表示已经
批准，也不替代后续的显式批准验证。当前证据必须支持那个具体操作，候选不得把审批准备扩大成一般
道德、安全或内容判断。

task_resolution_request 是当前缺少证据或有界工作时唯一的通用 resolver。它不让规划者选择
local、public、coding 或 text/computation specialist，也不让规划者选择时限、持久化、队列、租约、
工具参数、文件路径或最终措辞。后续 task orchestrator 和 specialist 依据其各自的公开合同处理这些
边界。现有 coding run 的批准和生命周期动作仍由其明确 action affordance 处理，不属于本阶段。

本阶段只判断未解决的证据需求与能力匹配，不改写请求、不选择最终对话，也不判断角色意愿、文笔
或虚构其他能力。

# 输出格式
只返回一个 JSON 对象，且字段必须恰好是 decisions。decisions 是一个 JSON 对象，键必须恰好
覆盖提供的 candidate handle，值必须是布尔值；true 表示核准，false 表示拒绝。候选项不得遗漏
或增添，只输出 JSON。
'''

__all__ = [
    "ACTION_AUTHORIZATION_PROMPT",
    "ACTION_AUTHORIZATION_PROMPT_CAP",
    "RESOLVER_AUTHORIZATION_PROMPT",
    "RESOLVER_AUTHORIZATION_PROMPT_CAP",
    "validate_authorization_decisions",
]
