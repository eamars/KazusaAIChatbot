"""Independent scene and event producers for conversation continuity."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    ConversationProgressContractError,
    compose_recorder_delta,
    event_handle_map,
    normalize_event_observation_bounds,
    normalize_scene_observation_bounds,
    source_handle_map,
    validate_event_observation_batch,
    validate_scene_observation,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationLogicalTurnV1,
    ConversationProgressEventUpdateV2,
    ConversationProgressEventV2,
    ConversationProgressRecorderDeltaV2,
    ConversationProgressRecordInput,
    ConversationProgressSceneUpdateV2,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_EPISODE_NARRATIVE_CHARS,
    MAX_INTERACTION_RECORDER_CHARS,
    MAX_RECORDER_HUMAN_PAYLOAD_CHARS,
    MAX_SCENE_RECORDER_HUMAN_PAYLOAD_CHARS,
    MAX_THREAD_FIELD_CHARS,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from kazusa_ai_chatbot.utils import log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)


_SCENE_RECORDER_PROMPT = '''\
你是角色短期对话连续性的场景观察者。你的唯一任务是描述本轮实际回应结束后已经成立的场景事实。
事件识别、事件生命周期、来源、存储、容量、压缩和下一轮目标都由其他边界负责。

# 输入语义
- `semantic_context.character_name` 是当前角色的准确名字。
- `semantic_context.current_local_time` 是本轮语义判断使用的当地时间，不是来源时间戳。
- `prior_scene` 是上一份已验证场景；为空表示这是第一份场景。
- `recent_turns` 是按时间排列的近期完整对话轮；`speaker_kind` 区分用户与角色，
  `speaker_name` 给出可用于事实描述的名字。
- `accepted_turn.current_input` 是当前用户输入。
- `accepted_turn.final_dialog` 是角色实际发出的回应。
- `accepted_turn.content_plan`、`logical_stance` 和 `character_intent` 只解释该实际回应的语义。

# 生成步骤
1. 只根据已经发生的输入和实际回应更新场景。
2. `scene_relation` 报告相对先前场景的关系：
   - `same`：同一场景继续。
   - `related`：相关转移，仍承接先前事实。
   - `new`：明显转场。
3. `episode_change` 只报告整段互动变化：
   - `none`：没有新的暂停、结束或恢复。
   - `paused`：整段互动明确暂停。
   - `finished`：整段互动明确结束。
   - `resumed`：先前暂停或结束的互动明确恢复。
4. 场景字段只写已成立事实。不得提出下一步行动、候选选择、未来目标或台词。
5. `overused_moves` 只记录本轮证据已经显示的重复回应模式。
6. 描述当前角色时，只能使用 `semantic_context.character_name` 的准确名字，或省略主语。
   不得用 assistant、bot、AI、角色等机器身份标签替代这个名字。
7. 对依赖今天、明天、下周等相对时间的约定、期限、计划或其他可执行事实，
   使用 `semantic_context.current_local_time` 换算为 `YYYY-MM-DD` 或
   `YYYY-MM-DD HH:MM`。无法唯一换算时省略该时间依赖事实，不保留相对时间表达。

# 字段约束
- `episode_narrative` 最多 900 字。
- 其余场景文本字段最多 240 字。
- 自由文本使用简体中文；专名、代码和输入原文保持原样。

# 输出格式
字段如下：
{
  "scene_relation": "same",
  "episode_change": "none",
  "episode_narrative": "",
  "current_thread": "",
  "character_stance": "",
  "user_goal": "",
  "current_blocker": "",
  "emotional_trajectory": "",
  "overused_moves": []
}
'''

_scene_recorder_llm = LLInterface()
_scene_recorder_llm_config = LLMCallConfig(
    stage_name=f'{__name__}.scene',
    route_name='CONSOLIDATION_LLM',
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.0,
    top_p=0.75,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


class ConversationProgressSceneOutputError(
    ConversationProgressContractError,
):
    """The scene producer failed its one-attempt contract."""


@dataclass(frozen=True)
class _SceneInvocation:
    """Validated scene output and per-owner telemetry."""

    scene: ConversationProgressSceneUpdateV2
    human_payload_chars: int
    provider_usage: dict[str, object]
    bound_normalizations: tuple[dict[str, object], ...]


async def _record_scene(
    record_input: ConversationProgressRecordInput,
) -> _SceneInvocation:
    """Invoke the scene-only producer once."""

    payload = build_scene_recorder_human_payload(record_input)
    human_json = _serialize_payload(payload)
    payload_chars = len(human_json)
    if payload_chars > MAX_SCENE_RECORDER_HUMAN_PAYLOAD_CHARS:
        raise ConversationProgressContextLimitError(
            'required scene-recorder context exceeds its hard character cap',
            owner='scene',
        )
    response = await _invoke_owner(
        llm=_scene_recorder_llm,
        config=_scene_recorder_llm_config,
        prompt=_SCENE_RECORDER_PROMPT,
        human_json=human_json,
        error_type=ConversationProgressSceneOutputError,
        owner='scene',
    )
    normalizations: tuple[dict[str, object], ...] = ()
    try:
        parsed = parse_llm_json_output(
            response.content,
            deterministic_only=True,
        )
        if not isinstance(parsed, Mapping):
            raise TypeError('scene semantic output must be an object')
        parsed = dict(parsed)
        if 'schema_version' in parsed:
            raise ValueError(
                'scene observation schema_version is code-owned'
            )
        parsed['schema_version'] = (
            'conversation_progress_scene_observation.v2'
        )
        parsed, normalizations = normalize_scene_observation_bounds(parsed)
        scene = validate_scene_observation(
            parsed,
            record_input=record_input,
        )
    except (TypeError, ValueError) as exc:
        raise ConversationProgressSceneOutputError(
            f'scene semantic output is invalid: {exc}'
        ) from exc
    return _SceneInvocation(
        scene=scene,
        human_payload_chars=payload_chars,
        provider_usage=_provider_usage(response),
        bound_normalizations=normalizations,
    )


_EVENT_RECORDER_PROMPT = '''\
你是角色短期连续性的事件核对者。你的唯一任务是逐项核对既有事件，并报告本轮新成立的独立事件。
场景摘要、整段互动状态、未来目标、存储、来源对象、时间、容量和压缩都由其他边界负责。

# 输入语义
- `semantic_context.character_name` 是当前角色的准确名字。
- `semantic_context.current_local_time` 是本轮语义判断使用的当地时间，不是来源时间戳。
- `prior_events` 中每个 `event_handle` 只在本次输入内指向一个具体既有事件。
- 每个既有事件的 `lifecycle_fact` 和 `relevance_fact` 是普通语义描述。
- `source_turns` 中每个 `source_handle` 指向一段可引用证据；`speaker_kind`
  区分用户与角色，`speaker_name` 给出可用于事实描述的名字。
- `accepted_turn.current_input` 是当前用户输入，固定句柄是 `current_input`。
- `accepted_turn.final_dialog` 是角色实际发出的回应，固定句柄是 `current_response`。
- 当前用户输入对用户作出的选择、拒绝、纠正和承诺，以及用户对自己执行事项的
  明确完成或停止声明具有最高权威。

# 核对步骤
1. 对 `prior_events` 中每个句柄恰好输出一行，并保持输入顺序：
   - 本轮没有改变该事件时，输出 `observation="unchanged"`。
   - 本轮改变该事件时，输出 `observation="changed"` 和完整变化事实。
   不得省略、重复或发明既有句柄。
2. 既有事件变化只更新摘要、结果、生命周期、相关性和来源。行动者、动作、对象、受益者、
   前提和义务方向由程序从已验证定义中保留。
3. 新事件只写入 `new_events`。每个新事件必须能独立完成、拒绝或被替代，并明确填写：
   - `actor`：行动者；
   - `action`：动作或决定；
   - `object`：被处理、选择、评价或作用的具体事项；
   - `beneficiary`：受益者，不适用时为空；
   - `precondition`：成立前提，不适用时为空。
   同一具体动作和对象内的方式、强度、阶段或反应变化更新原事件；
   同一类动作作用于不同具体部位或对象时，属于不同的独立事件；
   转向另一个可独立完成的动作或对象时另建新事件。
   前一具体部位已经得到评价，回应随后提出或开始另一个具体部位时，
   前一事件写 `concluded`，后一部位另建事件；整段同类互动继续不合并
   这两个事件。
4. `semantic_summary` 必须脱离上下文也能区分具体事件，不能只写序号、代词或此前事项。
5. 每个变化或新事件引用一个或多个已提供的 `source_turn_handles`。
6. `lifecycle_change` 只报告本轮变化：
   - `none`：既有状态不变；新事件尚未开始。
   - `began`：具体事件已经开始。
   - `concluded`：具体事件已完成、明确停止或得到确定结论。
   - `declined`：具体事件被明确拒绝。
   - `replaced`：具体事件被另一事项替代。
   - `reopened`：当前证据明确重开一个先前终结的既有事件。
   生命周期按每个具体事件判断，不按整段互动、总体目标、满意度或奖励是否完成判断。
   当前用户明确声明自己已经完成或停止与某个既有事件匹配的动作时，
   该事件写 `concluded`；总体满意度、奖励条件或整段互动继续不改变这个完成事实。
   既有事件已经得到明确结果或评价，且实际回应转向另一个可独立完成的事件时，
   前者写 `concluded`；整段互动继续不延长前者生命周期。
   如果来源对既有事件给出了明确结果、评价或停止信号，而实际回应已经
   转向另一个独立的动作、对象或部位，必须写 `concluded`，不能因为场景
   仍为 `same`、总体互动仍在继续或还有后续奖励条件而写 `none`。
   此时该句柄属于本轮已变化事件，先输出 `observation="changed"`，再在
   `lifecycle_change` 中写 `concluded`；`observation="unchanged"` 只用于
   本轮没有新事实的句柄。
   只有暂定且仍会在同一个具体事件尝试中变化的评价不算 `concluded`。
7. `relevance` 只报告当前语义作用：
   - `decision`：仍直接约束当前判断或区分已处理与未处理事项。
   - `scene`：帮助理解当前场景，但不直接约束判断。
   - `history`：仅为短期背景。
   终结状态本身不等于 `history`。当前轮要求选择或开始下一个事项，且某个已终结
   既有事件的结果用于区分已处理与未处理选项时，该事件仍写 `decision`。
8. 只报告已经成立的事实。不得提出下一步行动、候选选择、未来目标或台词。
9. 描述当前角色时，只能使用 `semantic_context.character_name` 的准确名字，或省略主语。
   不得用 assistant、bot、AI、角色等机器身份标签替代这个 `actor` 名字。
10. 对依赖今天、明天、下周等相对时间的约定、期限、计划或其他可执行事实，
    使用 `semantic_context.current_local_time` 换算为 `YYYY-MM-DD` 或
    `YYYY-MM-DD HH:MM`。无法唯一换算时不建立该时间依赖事件，不保留相对时间表达。

# 输出格式
顶层对象必须完整包含且仅包含 `existing_events` 和 `new_events` 两个字段。
两个数组即使为空也必须显式输出；没有新事件时必须原样输出 `"new_events": []`。
既有事件的 `observation` 只能是 `unchanged` 或 `changed`；它不是生命周期值。
`lifecycle_change` 只能是 `none`、`began`、`concluded`、`declined`、`replaced`
或 `reopened`，绝不能填写 `changed`。
`observation` 绝不能填写 `concluded`；完成事件的固定组合是
`"observation": "changed"` 与 `"lifecycle_change": "concluded"`。
未变化的既有事件必须完整包含且仅包含 `event_handle` 和 `observation`，严格为：
{
  "event_handle": "e1",
  "observation": "unchanged"
}
变化的既有事件必须完整包含且仅包含 `event_handle`、`observation`、`semantic_summary`、
`outcome`、`lifecycle_change`、`relevance` 和 `source_turn_handles`；
所有字段都必须显式输出，允许为空的文本也写空字符串：
{
  "event_handle": "e1",
  "observation": "changed",
  "semantic_summary": "",
  "outcome": "",
  "lifecycle_change": "none",
  "relevance": "scene",
  "source_turn_handles": ["current_input"]
}
既有事件完成时，仍然使用 `observation="changed"`，并单独写
`lifecycle_change="concluded"`：
{
  "event_handle": "e1",
  "observation": "changed",
  "semantic_summary": "具体事件已经完成",
  "outcome": "已得到确定结论",
  "lifecycle_change": "concluded",
  "relevance": "decision",
  "source_turn_handles": ["current_response"]
}
新事件严格为：
{
  "semantic_summary": "",
  "is_obligation": false,
  "actor": "",
  "action": "",
  "object": "",
  "beneficiary": "",
  "precondition": "",
  "outcome": "",
  "lifecycle_change": "none",
  "relevance": "scene",
  "source_turn_handles": ["current_input"]
}
顶层严格为：
{
  "existing_events": [],
  "new_events": []
}

# 返回前检查
1. 顶层恰好包含 `existing_events` 和 `new_events`。
2. `existing_events` 和 `new_events` 每次都输出为数组；没有项目也输出 `[]`。
3. `observation="unchanged"` 行恰好包含 `event_handle` 和 `observation`。
'''

_event_recorder_llm = LLInterface()
_event_recorder_llm_config = LLMCallConfig(
    stage_name=f'{__name__}.events',
    route_name='CONSOLIDATION_LLM',
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.0,
    top_p=0.75,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


class ConversationProgressEventOutputError(
    ConversationProgressContractError,
):
    """The event producer failed its one-attempt contract."""


@dataclass(frozen=True)
class _EventRecorderContext:
    """Model payload and exact private coverage domains."""

    payload: dict[str, object]
    event_handles: frozenset[str]
    source_handles: frozenset[str]


@dataclass(frozen=True)
class _EventInvocation:
    """Validated event output and per-owner telemetry."""

    event_updates: tuple[ConversationProgressEventUpdateV2, ...]
    human_payload_chars: int
    provider_usage: dict[str, object]
    bound_normalizations: tuple[dict[str, object], ...]


async def _record_events(
    record_input: ConversationProgressRecordInput,
) -> _EventInvocation:
    """Invoke the exact-coverage event producer once."""

    context = build_event_recorder_context(record_input)
    human_json = _serialize_payload(context.payload)
    payload_chars = len(human_json)
    if payload_chars > MAX_RECORDER_HUMAN_PAYLOAD_CHARS:
        raise ConversationProgressContextLimitError(
            'required event-recorder context exceeds its hard character cap',
            owner='event',
        )
    response = await _invoke_owner(
        llm=_event_recorder_llm,
        config=_event_recorder_llm_config,
        prompt=_EVENT_RECORDER_PROMPT,
        human_json=human_json,
        error_type=ConversationProgressEventOutputError,
        owner='event',
    )
    normalizations: tuple[dict[str, object], ...] = ()
    try:
        parsed = parse_llm_json_output(
            response.content,
            deterministic_only=True,
        )
        if not isinstance(parsed, Mapping):
            raise TypeError('event semantic output must be an object')
        parsed = dict(parsed)
        if 'schema_version' in parsed:
            raise ValueError(
                'event observation schema_version is code-owned'
            )
        parsed['schema_version'] = (
            'conversation_progress_event_observation_batch.v2'
        )
        parsed, normalizations = normalize_event_observation_bounds(parsed)
        updates = validate_event_observation_batch(
            parsed,
            record_input=record_input,
            supplied_event_handles=set(context.event_handles),
            supplied_source_handles=set(context.source_handles),
        )
    except (TypeError, ValueError) as exc:
        raise ConversationProgressEventOutputError(
            f'event semantic output is invalid: {exc}'
        ) from exc
    return _EventInvocation(
        event_updates=tuple(updates),
        human_payload_chars=payload_chars,
        provider_usage=_provider_usage(response),
        bound_normalizations=normalizations,
    )


class ConversationProgressContextLimitError(ValueError):
    """Required semantic recorder context exceeds an approved hard cap."""

    def __init__(
        self,
        message: str,
        *,
        owner: str,
        recorder_call_count: int = 0,
        event_attempt_count: int = 0,
        scene_attempt_count: int = 0,
        event_disposition: str = 'not_called',
        scene_disposition: str = 'not_called',
    ) -> None:
        """Retain exact producer-attempt telemetry on a preflight failure."""

        super().__init__(message)
        self.owner = owner
        self.recorder_call_count = recorder_call_count
        self.event_attempt_count = event_attempt_count
        self.scene_attempt_count = scene_attempt_count
        self.event_disposition = event_disposition
        self.scene_disposition = scene_disposition


class ConversationProgressRecorderOutputError(
    ConversationProgressContractError,
):
    """Authoritative event reconciliation failed closed."""

    def __init__(
        self,
        message: str,
        *,
        recorder_call_count: int,
        event_attempt_count: int,
        scene_attempt_count: int,
        event_disposition: str,
        scene_disposition: str,
    ) -> None:
        """Retain exact producer-attempt telemetry on event failure."""

        super().__init__(message)
        self.recorder_call_count = recorder_call_count
        self.event_attempt_count = event_attempt_count
        self.scene_attempt_count = scene_attempt_count
        self.event_disposition = event_disposition
        self.scene_disposition = scene_disposition


@dataclass(frozen=True)
class RecorderInvocationResult:
    """Combined validated observations with per-owner telemetry."""

    delta: ConversationProgressRecorderDeltaV2
    recorder_call_count: int
    event_attempt_count: int
    scene_attempt_count: int
    event_disposition: str
    scene_disposition: str
    event_human_payload_chars: int
    scene_human_payload_chars: int
    provider_usage: dict[str, object]
    bound_normalizations: tuple[dict[str, object], ...] = ()


async def record_with_llm(
    record_input: ConversationProgressRecordInput,
) -> RecorderInvocationResult:
    """Run the two independent post-turn producers concurrently."""

    scene_result, event_result = await asyncio.gather(
        _record_scene(record_input),
        _record_events(record_input),
        return_exceptions=True,
    )
    (
        scene,
        scene_disposition,
        scene_attempt_count,
        scene_payload_chars,
        scene_usage,
        scene_normalizations,
    ) = _resolve_scene_result(
        record_input=record_input,
        scene_result=scene_result,
    )
    if isinstance(event_result, ConversationProgressEventOutputError):
        raise ConversationProgressRecorderOutputError(
            str(event_result),
            recorder_call_count=scene_attempt_count + 1,
            event_attempt_count=1,
            scene_attempt_count=scene_attempt_count,
            event_disposition='failed_contract_or_provider',
            scene_disposition=scene_disposition,
        ) from event_result
    if isinstance(event_result, ConversationProgressContextLimitError):
        raise ConversationProgressContextLimitError(
            str(event_result),
            owner='event',
            recorder_call_count=scene_attempt_count,
            event_attempt_count=0,
            scene_attempt_count=scene_attempt_count,
            event_disposition='context_limit',
            scene_disposition=scene_disposition,
        ) from event_result
    if isinstance(event_result, BaseException):
        raise event_result

    delta = compose_recorder_delta(
        scene_observation=scene,
        event_updates=event_result.event_updates,
    )
    normalizations = tuple([
        *(
            {
                **normalization,
                'owner': 'scene',
            }
            for normalization in scene_normalizations
        ),
        *(
            {
                **normalization,
                'owner': 'event',
            }
            for normalization in event_result.bound_normalizations
        ),
    ])
    return RecorderInvocationResult(
        delta=delta,
        recorder_call_count=scene_attempt_count + 1,
        event_attempt_count=1,
        scene_attempt_count=scene_attempt_count,
        event_disposition='accepted',
        scene_disposition=scene_disposition,
        event_human_payload_chars=event_result.human_payload_chars,
        scene_human_payload_chars=scene_payload_chars,
        provider_usage={
            'event': event_result.provider_usage,
            'scene': scene_usage,
        },
        bound_normalizations=normalizations,
    )


def build_event_recorder_context(
    record_input: ConversationProgressRecordInput,
) -> _EventRecorderContext:
    """Fit event-only context and retain the exact supplied handle domains."""

    prior_event_map = event_handle_map(record_input)
    authoritative_event_handles = tuple(prior_event_map)
    source_map = source_handle_map(record_input)
    prior_events = [
        _prior_event_projection(handle, event)
        for handle, event in prior_event_map.items()
    ]
    source_turns = [
        _recorder_turn_projection(
            turn,
            source_handle=f't{index}',
            character_name=record_input['character_name'],
        )
        for index, turn in enumerate(
            record_input['interaction_logical_turns'],
            start=1,
        )
        if f't{index}' in source_map
    ]
    payload: dict[str, object] = {
        'semantic_context': _recorder_semantic_context(record_input),
        'prior_events': prior_events,
        'source_turns': source_turns,
        'accepted_turn': {
            'current_input': {
                'source_handle': (
                    'current_input'
                    if 'current_input' in source_map
                    else ''
                ),
                'text': record_input['decontextualized_input'],
            },
            'turn_outcome': record_input['turn_outcome'],
            'final_dialog': {
                'source_handle': (
                    'current_response'
                    if 'current_response' in source_map
                    else ''
                ),
                'fragments': list(record_input['final_dialog']),
            },
        },
    }
    _fit_event_payload(payload)
    projected_prior_events = payload['prior_events']
    if not isinstance(projected_prior_events, list):
        raise ConversationProgressContractError(
            'event payload must preserve the complete prior-event ledger'
        )
    projected_event_handles: list[str] = []
    for row in projected_prior_events:
        if not isinstance(row, Mapping):
            raise ConversationProgressContractError(
                'event payload must preserve the complete prior-event ledger'
            )
        handle = row.get('event_handle')
        if not isinstance(handle, str):
            raise ConversationProgressContractError(
                'event payload must preserve the complete prior-event ledger'
            )
        projected_event_handles.append(handle)
    if tuple(projected_event_handles) != authoritative_event_handles:
        raise ConversationProgressContractError(
            'event payload must preserve the complete prior-event ledger'
        )
    supplied_sources = {
        str(row['source_handle'])
        for row in payload['source_turns']
        if isinstance(row, Mapping)
    }
    accepted_turn = payload['accepted_turn']
    if not isinstance(accepted_turn, Mapping):
        raise TypeError('accepted_turn must be a mapping')
    for field_name in ('current_input', 'final_dialog'):
        value = accepted_turn[field_name]
        if not isinstance(value, Mapping):
            raise TypeError(f'accepted_turn.{field_name} must be a mapping')
        handle = value['source_handle']
        if isinstance(handle, str) and handle:
            supplied_sources.add(handle)
    return _EventRecorderContext(
        payload=payload,
        event_handles=frozenset(authoritative_event_handles),
        source_handles=frozenset(supplied_sources),
    )


def build_scene_recorder_human_payload(
    record_input: ConversationProgressRecordInput,
) -> dict[str, object]:
    """Build bounded scene-only dynamic context."""

    prior_packet = record_input['prior_episode_state']
    prior_scene: dict[str, object] | None = None
    if prior_packet is not None:
        prior_scene = {
            'episode_progress_fact': _prior_episode_progress_fact(
                prior_packet['status']
            ),
            'scene_relation_fact': _prior_scene_relation_fact(
                prior_packet['continuity']
            ),
            'episode_narrative': prior_packet['episode_narrative'],
            'current_thread': prior_packet['current_thread'],
            'character_stance': prior_packet['character_stance'],
            'user_goal': prior_packet['user_goal'],
            'current_blocker': prior_packet['current_blocker'],
            'emotional_trajectory': prior_packet['emotional_trajectory'],
            'overused_moves': list(prior_packet['overused_moves']),
        }
    recent_turns = [
        _recorder_turn_projection(
            turn,
            source_handle=f't{index}',
            character_name=record_input['character_name'],
        )
        for index, turn in enumerate(
            record_input['interaction_logical_turns'][-4:],
            start=1,
        )
    ]
    payload: dict[str, object] = {
        'semantic_context': _recorder_semantic_context(record_input),
        'prior_scene': prior_scene,
        'recent_turns': recent_turns,
        'accepted_turn': {
            'current_input': record_input['decontextualized_input'][
                :MAX_INTERACTION_RECORDER_CHARS
            ],
            'turn_outcome': record_input['turn_outcome'],
            'content_plan': dict(record_input['content_plan']),
            'logical_stance': record_input['logical_stance'],
            'character_intent': record_input['character_intent'],
            'final_dialog': list(record_input['final_dialog']),
        },
    }
    projected_turns = payload['recent_turns']
    if not isinstance(projected_turns, list):
        raise TypeError('scene recent_turns must be a list')
    while (
        _payload_chars(payload) > MAX_SCENE_RECORDER_HUMAN_PAYLOAD_CHARS
        and projected_turns
    ):
        projected_turns.pop(0)
    if _payload_chars(payload) > MAX_SCENE_RECORDER_HUMAN_PAYLOAD_CHARS:
        raise ConversationProgressContextLimitError(
            'required scene-recorder context exceeds its hard character cap',
            owner='scene',
        )
    return payload


def render_scene_recorder_prompt() -> str:
    """Return the static scene-owner prompt for inspection."""

    return _SCENE_RECORDER_PROMPT


def render_event_recorder_prompt() -> str:
    """Return the static event-owner prompt for inspection."""

    return _EVENT_RECORDER_PROMPT


def _fit_event_payload(
    payload: dict[str, object],
) -> None:
    """Drop older turn text while retaining the complete prior ledger."""

    if _payload_chars(payload) <= MAX_RECORDER_HUMAN_PAYLOAD_CHARS:
        return
    source_turns = payload['source_turns']
    if not isinstance(source_turns, list):
        raise TypeError('event source_turns must be a list')
    while (
        _payload_chars(payload) > MAX_RECORDER_HUMAN_PAYLOAD_CHARS
        and source_turns
    ):
        source_turns.pop(0)
    if _payload_chars(payload) > MAX_RECORDER_HUMAN_PAYLOAD_CHARS:
        raise ConversationProgressContextLimitError(
            'required event-recorder context exceeds its hard character cap',
            owner='event',
        )


def _prior_event_projection(
    event_handle: str,
    event: ConversationProgressEventV2,
) -> dict[str, object]:
    """Project one prior event without storage identity or timestamps."""

    return {
        'event_handle': event_handle,
        'semantic_summary': event['semantic_summary'],
        'is_obligation': event['is_obligation'],
        'actor': event['actor'],
        'action': event['action'],
        'object': event['object'],
        'beneficiary': event['beneficiary'],
        'precondition': event['precondition'],
        'outcome': event['outcome'],
        'lifecycle_fact': _prior_lifecycle_fact(event['state']),
        'relevance_fact': _prior_relevance_fact(event['retention']),
    }


def _recorder_turn_projection(
    turn: ConversationLogicalTurnV1,
    *,
    source_handle: str,
    character_name: str,
) -> dict[str, object]:
    """Project one logical turn as semantic text and a short handle."""

    joined_text = ' '.join(turn['fragments'])
    role = turn['role']
    if role == 'assistant':
        speaker_kind = 'character'
        speaker_name = character_name
    elif role == 'user':
        speaker_kind = 'user'
        speaker_name = turn['display_name'].strip() or 'current user'
    else:
        raise ValueError('recorder logical-turn role is invalid')
    return {
        'source_handle': source_handle,
        'speaker_kind': speaker_kind,
        'speaker_name': speaker_name,
        'text': joined_text[:MAX_INTERACTION_RECORDER_CHARS].rstrip(),
    }


def _recorder_semantic_context(
    record_input: ConversationProgressRecordInput,
) -> dict[str, str]:
    """Project identity and clock context without exposing source metadata."""

    character_name = record_input['character_name']
    if not isinstance(character_name, str) or not character_name.strip():
        raise ValueError('recorder character_name is required')
    current_local_time = format_storage_utc_for_llm(
        record_input['storage_timestamp_utc']
    )
    if not current_local_time:
        raise ValueError('recorder semantic current_local_time is required')
    return {
        'character_name': character_name.strip(),
        'current_local_time': current_local_time,
    }


def _preserved_or_initial_scene(
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressSceneUpdateV2:
    """Preserve validated scene facts or seed them from accepted semantics."""

    prior_packet = record_input['prior_episode_state']
    if prior_packet is not None:
        return {
            'continuity': prior_packet['continuity'],
            'status': prior_packet['status'],
            'episode_narrative': prior_packet['episode_narrative'],
            'current_thread': prior_packet['current_thread'],
            'character_stance': prior_packet['character_stance'],
            'user_goal': prior_packet['user_goal'],
            'current_blocker': prior_packet['current_blocker'],
            'emotional_trajectory': prior_packet['emotional_trajectory'],
            'overused_moves': list(prior_packet['overused_moves']),
        }
    content_plan = record_input['content_plan']
    semantic_content = content_plan['semantic_content']
    current_input = record_input['decontextualized_input']
    logical_stance = record_input['logical_stance']
    for field_name, value in (
        ('content_plan.semantic_content', semantic_content),
        ('decontextualized_input', current_input),
        ('logical_stance', logical_stance),
    ):
        if not isinstance(value, str):
            raise TypeError(f'{field_name} must be text')
    narrative = semantic_content or current_input
    return {
        'continuity': 'same_episode',
        'status': 'active',
        'episode_narrative': narrative[:MAX_EPISODE_NARRATIVE_CHARS],
        'current_thread': current_input[:MAX_THREAD_FIELD_CHARS],
        'character_stance': logical_stance[:MAX_THREAD_FIELD_CHARS],
        'user_goal': current_input[:MAX_THREAD_FIELD_CHARS],
        'current_blocker': '',
        'emotional_trajectory': '',
        'overused_moves': [],
    }


def _resolve_scene_result(
    *,
    record_input: ConversationProgressRecordInput,
    scene_result: _SceneInvocation | BaseException,
) -> tuple[
    ConversationProgressSceneUpdateV2,
    str,
    int,
    int,
    dict[str, object],
    tuple[dict[str, object], ...],
]:
    """Resolve the lower-authority scene lane and exact call telemetry."""

    if isinstance(scene_result, ConversationProgressSceneOutputError):
        attempt_count = 1
    elif isinstance(scene_result, ConversationProgressContextLimitError):
        attempt_count = 0
    elif isinstance(scene_result, BaseException):
        raise scene_result
    else:
        return (
            scene_result.scene,
            'accepted',
            1,
            scene_result.human_payload_chars,
            scene_result.provider_usage,
            scene_result.bound_normalizations,
        )

    scene = _preserved_or_initial_scene(record_input)
    disposition = (
        'preserved_prior'
        if record_input['prior_episode_state'] is not None
        else 'initialized_from_accepted_turn'
    )
    logger.warning(
        'Conversation progress scene observer degraded: '
        f'disposition={disposition} '
        f'error={type(scene_result).__name__}'
    )
    return (
        scene,
        disposition,
        attempt_count,
        0,
        {'status': 'unavailable'},
        (),
    )


async def _invoke_owner(
    *,
    llm: LLInterface,
    config: LLMCallConfig,
    prompt: str,
    human_json: str,
    error_type: type[ConversationProgressContractError],
    owner: str,
) -> object:
    """Invoke one producer and translate only known provider failures."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_json),
    ]
    try:
        response = await llm.ainvoke(messages, config=config)
    except (
        OpenAIError,
        httpx.HTTPError,
        ConnectionError,
        OSError,
        RuntimeError,
        TimeoutError,
    ) as exc:
        raise error_type(f'{owner} provider call failed') from exc
    logger.debug(
        f'Conversation progress {owner} input: chars={len(human_json)} '
        f'payload={log_preview(human_json)}'
    )
    return response


def _prior_episode_progress_fact(status: str) -> str:
    """Translate persisted episode status into plain model context."""

    return {
        'active': '这一段互动仍在继续',
        'suspended': '这一段互动已经暂停，等待明确恢复',
        'closed': '这一段互动已经结束',
    }[status]


def _prior_scene_relation_fact(continuity: str) -> str:
    """Translate persisted continuity into plain model context."""

    return {
        'same_episode': '上一轮仍在同一段场景中',
        'related_shift': '上一轮发生了相关转移',
        'sharp_transition': '上一轮发生了明显转场',
    }[continuity]


def _prior_lifecycle_fact(state: str) -> str:
    """Translate one persisted lifecycle enum into plain model context."""

    return {
        'open': '尚未开始，仍待处理',
        'in_progress': '已经开始，尚未得到确定结论',
        'completed': '已经完成或得到确定结论',
        'rejected': '已经明确拒绝',
        'superseded': '已经被另一事项替代',
    }[state]


def _prior_relevance_fact(retention: str) -> str:
    """Translate one persisted retention enum into plain model context."""

    return {
        'decision_critical': '仍直接约束当前判断',
        'active_scene': '仍帮助理解当前场景',
        'background': '只作为短期背景',
    }[retention]


def _serialize_payload(payload: Mapping[str, object]) -> str:
    """Serialize one model payload deterministically."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )


def _payload_chars(payload: Mapping[str, object]) -> int:
    """Measure one dynamic payload after final normalization."""

    return len(_serialize_payload(payload))


def _provider_usage(response: object) -> dict[str, object]:
    """Extract provider token telemetry when exposed."""

    usage = getattr(response, 'usage_metadata', None)
    if isinstance(usage, Mapping):
        return dict(usage)
    response_metadata = getattr(response, 'response_metadata', None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get('token_usage')
        if isinstance(token_usage, Mapping):
            return dict(token_usage)
    return {'status': 'unavailable'}


def _json_default(value: object) -> object:
    """Render supported dynamic payload values."""

    if hasattr(value, 'isoformat'):
        return value.isoformat()
    raise TypeError(
        f'unsupported recorder payload value: {type(value).__name__}'
    )
