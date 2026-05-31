"""Deterministic prompt-text contract checks for cognition stages."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l1 as l1_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2c2 as l2c2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2d as l2d_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


_AFFECTED_PROMPTS = (
    (
        'L1 subconscious',
        '_COGNITION_SUBCONSCIOUS_PROMPT',
        l1_module._COGNITION_SUBCONSCIOUS_PROMPT,
    ),
    (
        'L2a consciousness',
        '_COGNITION_CONSCIOUSNESS_PROMPT',
        l2_module._COGNITION_CONSCIOUSNESS_PROMPT,
    ),
    (
        'L2b boundary',
        '_BOUNDARY_CORE_PROMPT',
        l2_module._BOUNDARY_CORE_PROMPT,
    ),
    (
        'L2c1 judgment',
        '_JUDGEMENT_CORE_PROMPT',
        l2_module._JUDGEMENT_CORE_PROMPT,
    ),
    (
        'L2c2 social context',
        '_CONTEXTUAL_AGENT_PROMPT',
        l2c2_module._CONTEXTUAL_AGENT_PROMPT,
    ),
    (
        'L2d action initializer',
        '_ACTION_INITIALIZER_PROMPT',
        l2d_module._ACTION_INITIALIZER_PROMPT,
    ),
)
_SELF_COGNITION_SOURCE_TERMS = (
    '自我认知',
    '内部自我认知',
    '内部想法',
    '内部思考',
    'internal_thought',
    'internal_monologue',
)
_SELF_COGNITION_OWNERSHIP_TERMS = (
    '我自己的观察资料',
    '我的内部观察资料',
    '我刚看到或回顾的观察资料',
    '内部观察资料',
)
_REFLECTION_SOURCE_TERMS = (
    '反思资料',
    'reflection_artifact',
    'reflection_signal',
)
_REFLECTION_OWNERSHIP_TERMS = (
    '我自己的反思资料',
    '我的反思资料',
    '我的反思',
)
_EXTERNAL_SPEECH_TERMS = (
    '外部用户发言',
    '外部用户说话',
    '当前用户发言',
    '用户当前发言',
    '当前外部频道',
    '当前外部说话内容',
    '外部说话内容',
    '当前外部文本',
)
_NEGATION_TERMS = (
    '不是',
    '不得',
    '不能',
    '不要',
    '禁止',
)
_FORBIDDEN_RECLASSIFICATION_TERMS = (
    '用户输入',
    '用户提供',
    '用户发言',
    '用户说话',
    '当前群成员发言',
    '任何人正在对角色说话',
    '正在对角色说话',
)
_METADATA_COPY_VERBS = (
    '不要复制',
    '不得复制',
    '禁止复制',
    '不要照抄',
    '不得照抄',
    '禁止照抄',
    '不要搬运',
    '不得搬运',
)
_METADATA_SOURCE_CATEGORIES = (
    (
        '来源包标题',
        '源包标题',
        'source-packet heading',
        'source packet heading',
        '标题',
        '字段名',
    ),
    ('JSON', 'json'),
    ('时间戳', 'timestamp'),
    (
        '语义标签键',
        '语义标签字段',
        'semantic-label key',
        'semantic label key',
        'semantic_labels',
    ),
    ('传输摘要', 'transport summary'),
    ('模型可见元数据', '模型面对元数据', 'model-facing metadata'),
)
_GENERATED_FIELD_TERMS = (
    '生成字段',
    '自由文本字段',
    'internal_monologue',
    'judgment_note',
    'detail',
    'reason',
)
_FORBIDDEN_SELF_COGNITION_WORDING = (
    '自检',
    '需要接上',
    '数据身份：',
    '进入注意的原因：',
    '阅读方式',
)
_FORBIDDEN_THIRD_PERSON_SELF_REFERENCE_PHRASES = (
    '角色自己的反思资料',
    '角色自己的内部观察资料',
    '角色自己的观察资料',
    '角色刚看到',
    '角色是否',
    '有人把话题交给角色',
    '角色与被观察现场',
    '角色与现场',
    '角色判断描述',
    '角色如何理解',
    '角色已经沉淀',
    '触及角色的',
    '夺取角色',
    '提到角色',
    '对角色说话',
    '命令角色发言',
    '角色想做什么',
    '角色决定把话',
    '角色需要等待',
    '角色愿意把',
    '角色的主要动机',
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _assert_contains_any(
    prompt_name: str,
    prompt_text: str,
    terms: tuple[str, ...],
    contract: str,
) -> None:
    if _contains_any(prompt_text, terms):
        return

    pytest.fail(
        f'{prompt_name} must contain {contract}; accepted terms={terms!r}',
    )


def _assert_self_cognition_reclassification_rule(
    prompt_name: str,
    prompt_text: str,
) -> None:
    _assert_source_reclassification_rule(
        prompt_name=prompt_name,
        prompt_text=prompt_text,
        source_terms=_SELF_COGNITION_SOURCE_TERMS,
        source_label='self-cognition or internal-thought material',
    )


def _assert_reflection_reclassification_rule(
    prompt_name: str,
    prompt_text: str,
) -> None:
    _assert_source_reclassification_rule(
        prompt_name=prompt_name,
        prompt_text=prompt_text,
        source_terms=_REFLECTION_SOURCE_TERMS,
        source_label='reflection material',
    )


def _assert_source_reclassification_rule(
    *,
    prompt_name: str,
    prompt_text: str,
    source_terms: tuple[str, ...],
    source_label: str,
) -> None:
    compact_text = re.sub(r'\s+', '', prompt_text)
    source_pattern = '|'.join(re.escape(term) for term in source_terms)
    role_pattern = '|'.join(
        re.escape(term) for term in _FORBIDDEN_RECLASSIFICATION_TERMS
    )
    negation_pattern = '|'.join(re.escape(term) for term in _NEGATION_TERMS)
    rule_pattern = re.compile(
        rf'({source_pattern}).{{0,80}}({negation_pattern}).{{0,80}}'
        rf'({role_pattern})'
        rf'|({negation_pattern}).{{0,80}}({source_pattern}).{{0,80}}'
        rf'({role_pattern})'
        rf'|({source_pattern}).{{0,80}}({role_pattern}).{{0,80}}'
        rf'({negation_pattern})'
    )

    if rule_pattern.search(compact_text):
        return

    pytest.fail(
        f'{prompt_name} must explicitly forbid treating {source_label} as '
        'user input, user-provided content, or '
        'live user speech.',
    )


def test_affected_prompts_require_simplified_chinese_generated_free_text() -> None:
    """Generated free-text policy must remain visible in every stage prompt."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        assert '简体中文' in prompt_text, prompt_name
        _assert_contains_any(
            prompt_name,
            prompt_text,
            ('新生成', '新生成的', '由你新生成'),
            'a generated-text marker',
        )
        _assert_contains_any(
            prompt_name,
            prompt_text,
            ('自由文本字段', '内部自由文本字段'),
            'a free-text field marker',
        )


def test_affected_prompts_distinguish_external_speech_from_internal_observation() -> None:
    """Prompts must distinguish user speech from character-owned observations."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _EXTERNAL_SPEECH_TERMS,
            'external user speech wording',
        )
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _SELF_COGNITION_SOURCE_TERMS,
            'self-cognition or internal-thought source wording',
        )
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _SELF_COGNITION_OWNERSHIP_TERMS,
            'character-owned observation wording',
        )


def test_affected_prompts_distinguish_reflection_from_external_speech() -> None:
    """Prompts must distinguish reflection artifacts from live user speech."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _REFLECTION_SOURCE_TERMS,
            'reflection source wording',
        )
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _REFLECTION_OWNERSHIP_TERMS,
            'character-owned reflection wording',
        )
        _assert_reflection_reclassification_rule(prompt_name, prompt_text)


def test_affected_prompts_forbid_reclassifying_internal_thought_as_user_speech() -> None:
    """Internal self-cognition material must not become live user speech."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        _assert_self_cognition_reclassification_rule(prompt_name, prompt_text)


def test_affected_prompts_forbid_copying_source_packet_metadata() -> None:
    """Generated fields must summarize decisions instead of packet structure."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _METADATA_COPY_VERBS,
            'a metadata-copy prohibition',
        )
        for category_terms in _METADATA_SOURCE_CATEGORIES:
            _assert_contains_any(
                prompt_name,
                prompt_text,
                category_terms,
                'all source-packet metadata categories',
            )
        _assert_contains_any(
            prompt_name,
            prompt_text,
            _GENERATED_FIELD_TERMS,
                'generated-field targets for the metadata-copy prohibition',
        )


@pytest.mark.asyncio
async def test_l1_subconscious_payload_passes_character_state_in_human_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L1 runtime state belongs in Human JSON, not formatted system text."""

    turn_clock = build_turn_clock_from_storage_utc(
        '2026-05-25T09:30:00+00:00',
    )
    episode = build_text_chat_cognitive_episode(
        episode_id='prompt-contract-l1-episode',
        percept_id='prompt-contract-l1-percept',
        storage_timestamp_utc=turn_clock['storage_timestamp_utc'],
        local_time_context=turn_clock['local_time_context'],
        user_input='Is it safe to remove this prompt section?',
        platform='debug',
        platform_channel_id='debug-private-1',
        channel_type='private',
        platform_message_id='message-1',
        platform_user_id='user-1',
        global_user_id='global-user-1',
        user_name='Ran',
        active_turn_platform_message_ids=['message-1'],
        active_turn_conversation_row_ids=[],
        debug_modes={},
        target_addressed_user_ids=['character-1'],
        target_broadcast=False,
    )
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value=AIMessage(
                content=(
                    '{"emotional_appraisal":"先稳住",'
                    '"interaction_subtext":"在问风险"}'
                ),
            ),
        ),
    )
    monkeypatch.setattr(l1_module, '_subconscious_llm', fake_llm)
    state = {
        'character_profile': {
            'name': 'Kazusa',
            'mood': 'guarded',
            'global_vibe': 'quiet review',
            'personality_brief': {'mbti': 'INTJ'},
        },
        'user_profile': {
            'last_relationship_insight': 'The user expects precise tradeoffs.',
        },
        'cognitive_episode': episode,
        'user_input': 'Is it safe to remove this prompt section?',
        'indirect_speech_context': '',
    }

    result = await l1_module.call_cognition_subconscious(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert result['emotional_appraisal'] == '先稳住'
    assert '# 输入格式' not in system_prompt
    assert payload['character_state'] == {
        'mood': 'guarded',
        'global_vibe': 'quiet review',
        'last_relationship_insight': 'The user expects precise tradeoffs.',
    }


def test_l2d_prompt_defines_speak_and_scene_grounded_detail() -> None:
    """L2d must keep speak visible and detail grounded in the current scene."""
    prompt_text = l2d_module._ACTION_INITIALIZER_PROMPT

    assert '`speak`' in prompt_text
    _assert_contains_any(
        '_ACTION_INITIALIZER_PROMPT',
        prompt_text,
        ('可见文字回复', '可见外部频道文字', '当前外部频道'),
        '`speak` as visible external-channel text',
    )
    _assert_contains_any(
        '_ACTION_INITIALIZER_PROMPT',
        prompt_text,
        ('当前场景', '当前可见回复目标', '当前可见行动目标'),
        '`detail` as the current scene action target',
    )
    _assert_contains_any(
        '_ACTION_INITIALIZER_PROMPT',
        prompt_text,
        ('不要生成最终发言文本', '不是最终对话文本', '不得生成最终对话文本'),
        '`detail` not as final dialog text',
    )
    _assert_contains_any(
        '_ACTION_INITIALIZER_PROMPT',
        prompt_text,
        ('包标题', '时间戳', '传输摘要', '模型可见元数据'),
        '`detail` not as packet metadata',
    )


def test_l2d_prompt_preserves_resolver_terminal_boundaries() -> None:
    """L2d should keep resolver selection inside source and terminal limits."""
    prompt_text = l2d_module._ACTION_INITIALIZER_PROMPT

    for required_text in (
        '`approval_preparation`',
        '`human_clarification`',
        '`self_goal_resolution`',
        '触发来源是 `user_message`',
        '不是 resolver capability',
        'blocked observation',
        'pending resume',
        '不要再次请求同一个 blocked capability',
        '不要重复请求同类检索',
        '更窄、不同、未尝试的证据目标',
        'approval preview 必须能力扎根',
        '不得编造上下文没有提供的工具、权限、外部执行机制或验证机制',
        '时效性、公开来源绑定或用户明确要求核实来源的事实',
        '区分来源确认、角色推断和当前无法验证的部分',
        '来源类别、证据轨道或比较对象',
        '不得改写成跨来源一致、无冲突或已确认',
        '继续处理原始用户目标',
        '不要只确认“收到”就结束',
        '回答原始目标的可见回复目标',
        '而不是把用户补充信息当作新的独立闲聊',
        '主要交付部分',
        '不要只回答其中一个子问题后把必要交付推迟到下一轮',
        '给出完整的最佳努力结果、证据限制和必要步骤',
        '非必需偏好或排序口径',
        '解析器续轮原则',
        '返回一个 `speak` action_request',
        '最小缺口',
        '触发来源：user_message',
        '禁止返回 `self_goal_resolution`',
        '必须写入 `resolver_capability_requests`',
        '绝不能写入 `action_requests.capability`',
        '缺少可选范围、标准或排序口径不等于缺少必须由用户提供的信息',
        '没有本轮 `rag_evidence` observation',
        '不要直接选择 `speak` 跳过审批准备',
        '不得编造监控、校验、自动检查或外部执行能力',
        '不要继续换同义词重复搜索',
        '已有证据和一般判断完成的分析、决策、方案或排查任务',
        '当前外部断言',
        '可行动标准和最后核实步骤',
        '分析、决策、方案设计、风险清单或下一步行动',
        '不要为了给一般判断背书而启动 `web_evidence`',
        '必须有本轮 `web_evidence` observation 支撑后才能 `speak` 给出',
        '否则先请求 `web_evidence`',
        '只给阻塞说明、可行动标准和最后核实步骤',
        '需要先收束目标、整理优先级、拆解私有后续判断或形成下一步内部目标',
        'resolver_capability_requests[].capability_kind',
        '私有目标收束已经完成',
        '没有新的具体私有动作就返回空数组',
        '`resolver_goal_progress`',
        '语义进度表',
        '不得替换原始目标',
        '`pending`、`partial`、`satisfied`、`blocked`',
        '`resolver_goal_progress.final_response_requirements` 是 L3/dialog 的交付清单',
        '"schema_version": "resolver_goal_progress.v1"',
        '"final_response_requirements"',
    ):
        assert required_text in prompt_text


def test_l3_content_anchor_scope_preserves_complete_plan_deliverables() -> None:
    """Content anchors should leave enough room for multi-part answers."""

    prompt_text = l3_module._CONTENT_ANCHOR_AGENT_PROMPT

    for required_text in (
        '完整方案',
        '多候选推荐',
        '风险说明',
        '给足覆盖空间',
        '不要用过短篇幅迫使下游只回答一个子问题',
        '保留交付清单',
        '含 `原始目标`',
        '不能取代原始目标',
        '保留原始目标中的主要交付部分',
        '保留原始目标',
        '必须把原始目标和当前补充约束合并处理',
        '不得只回答当前补充约束或检索结果里最显眼的一部分',
        '可执行的时间切分和行动顺序',
        '尽量写成具体时段',
        '起点或待确认对象 -> 可公开核实的中间锚点 -> 结束点或回退点',
        '不得只写模糊方向',
        '不得只列候选或把计划收束成新的口味、预算、时间偏好追问',
        '证据边界',
        '不得把缺失证据改写成已确认事实',
        '来源类别边界',
        '多个来源类别、证据轨道或比较对象',
        '不得被改写成一致、无冲突或已确认',
        '当前事实防编造',
        '不要在 `[ANSWER]` 要求下游给出具体当前断言',
        '具体对象、属性、状态、时间和可用性',
        '必须来自 `[FACT]` 或 `rag_result.external_evidence`',
        '必须禁止下游给出具体断言',
        '泛化说明不得偷换成具体对象示例',
        '行动骨架和最终核实清单',
        '证据阻塞不是新的 HIL 澄清',
        '必须完成最佳努力阻塞答复',
        '不能把主回应改成新的追问或认可请求',
        '可调整条件只能作为可选退路',
        '终止型证据阻塞必须在当前可见回答内收束',
        '临时处理状态或延后承诺',
        '把已要求的主要交付推迟到下一轮',
        '不能发明当前事实',
        '250-500 字或多段短句',
        '角色口吻可以碎片化，但交付不能碎片化到不可用',
        '含 `目标进度` 或 `resolver_goal_progress`',
        '已清洗的 resolver observation 摘要',
        '该收束目标优先于 `character_intent = CLARIFY`',
        'deliverables、blockers 和 final_response_requirements',
        '不得只回答最新 evidence 或最显眼的一个子问题',
        'pending`、`partial` 和 `blocked` 的交付项',
        '新的澄清或认可请求',
    ):
        assert required_text in prompt_text


def test_prompts_preserve_structured_output_enums() -> None:
    """Prompt rewrites must keep downstream enum vocabularies visible."""
    for enum_value in ('CONFIRM', 'REFUSE', 'TENTATIVE', 'DIVERGE', 'CHALLENGE'):
        assert enum_value in l2_module._COGNITION_CONSCIOUSNESS_PROMPT
        assert enum_value in l2_module._JUDGEMENT_CORE_PROMPT

    for enum_value in (
        'PROVIDE',
        'BANTAR',
        'REJECT',
        'EVADE',
        'CONFRONT',
        'DISMISS',
        'CLARIFY',
    ):
        assert enum_value in l2_module._COGNITION_CONSCIOUSNESS_PROMPT
        assert enum_value in l2_module._JUDGEMENT_CORE_PROMPT

    for enum_value in (
        'none',
        'identity_override',
        'control_imposition',
        'authority_claim',
        'relational_distortion',
        'mixed',
        'allow',
        'guarded',
        'hesitant',
        'reject',
    ):
        assert enum_value in l2_module._BOUNDARY_CORE_PROMPT

    for capability in (
        'speak',
        'memory_lifecycle_update',
        'trigger_future_cognition',
    ):
        assert capability in l2d_module._ACTION_INITIALIZER_PROMPT


def test_prompts_do_not_use_forbidden_self_cognition_wording() -> None:
    """Source-aware prompts should avoid stale self-cognition framing terms."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        for forbidden_wording in _FORBIDDEN_SELF_COGNITION_WORDING:
            assert forbidden_wording not in prompt_text, (
                f'{prompt_name} contains forbidden wording: '
                f'{forbidden_wording}'
            )


def test_prompts_do_not_anchor_generated_self_text_to_third_person_role() -> None:
    """Prompt free-text guidance should not invite third-person self wording."""
    for _, prompt_name, prompt_text in _AFFECTED_PROMPTS:
        for forbidden_phrase in _FORBIDDEN_THIRD_PERSON_SELF_REFERENCE_PHRASES:
            assert forbidden_phrase not in prompt_text, (
                f'{prompt_name} contains third-person self-reference anchor: '
                f'{forbidden_phrase}'
            )
