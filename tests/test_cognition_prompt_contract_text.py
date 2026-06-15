"""Deterministic prompt-text contract checks for cognition stages."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.cognition_episode import build_text_chat_cognitive_episode
from kazusa_ai_chatbot.cognition_chain_core.stages import l1 as l1_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2 as l2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l2c2 as l2c2_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
    ACTION_ROUTER_PROMPT,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc
from llm_test_helpers import bind_test_llm


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
        'L2d action selection',
        'ACTION_ROUTER_PROMPT',
        ACTION_ROUTER_PROMPT,
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
    monkeypatch.setattr(l1_module, '_subconscious_llm', bind_test_llm(fake_llm, "subconscious_llm"))
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
    """L2d must keep visible-surface action semantics and detail grounded in
    the current scene.  The new prompt uses generic affordance terms instead
    of hardcoded capability names."""
    prompt_text = ACTION_ROUTER_PROMPT

    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('可见表面动作', '可见动作', 'action_affordances'),
        'visible-surface action concept (replaces hardcoded `speak`)',
    )
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('当前场景', '当前可见回复目标', '当前可见行动目标'),
        '`detail` as the current scene action target',
    )
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('不要生成最终发言文本', '不是最终发言文本', '不得生成最终对话文本'),
        '`detail` not as final dialog text',
    )
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('包标题', '时间戳', '传输摘要', '模型可见元数据'),
        '`detail` not as packet metadata',
    )


def test_l2d_prompt_preserves_resolver_terminal_boundaries() -> None:
    """L2d should keep resolver selection inside source and terminal limits.

    The new prompt uses generic affordance references instead of hardcoded
    capability names.  Assertions are grouped by the behavioral contract they
    guard so that failures point to the rule that regressed.
    """
    prompt_text = ACTION_ROUTER_PROMPT

    # -- affordance-driven capability selection (replaces hardcoded names) --
    for required_text in (
        'capabilities.resolver_affordances',
        'capabilities.action_affordances',
        'resolver_capability_requests[].capability_kind',
    ):
        assert required_text in prompt_text

    # -- source identification and trigger boundaries --
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('触发来源', 'trigger_source'),
        'trigger-source awareness',
    )
    assert 'user_message' in prompt_text

    # -- blocked / pending resume handling --
    assert 'blocked' in prompt_text
    assert 'pending_resolver_resume' in prompt_text
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('不要再次请求同一', '不要再次请求同一个 blocked'),
        'blocked capability dedup rule',
    )
    assert '不要重复请求同类检索' in prompt_text
    assert '更窄、不同、未尝试的证据目标' in prompt_text

    # -- approval and fabrication guardrails --
    assert 'approval preview 必须能力扎根' in prompt_text
    assert '不得编造上下文没有提供的工具、权限、外部执行机制或验证机制' in prompt_text
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('不得编造监控', '不得编造'),
        'no fabricated monitoring or execution capabilities',
    )
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('不要直接选择 `speak` 跳过审批准备',
         '跳过审批', '审批准备'),
        'no skipping approval preparation',
    )

    # -- evidence quality and source boundaries --
    assert '时效性、公开来源绑定或用户明确要求核实来源的事实' in prompt_text
    assert '区分来源确认、角色推断和当前无法验证的部分' in prompt_text
    assert '来源类别、证据轨道或比较对象' in prompt_text
    assert '不得改写成跨来源一致、无冲突或已确认' in prompt_text
    assert '当前外部断言' in prompt_text
    assert 'observation' in prompt_text

    # -- original goal continuity --
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('继续处理原始用户目标', '继续推进', 'original_goal'),
        'original goal continuity',
    )
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('不要只确认"收到"就结束', '不要只确认收到', '不要只确认'),
        'must not just acknowledge receipt',
    )
    assert '回答原始目标的可见回复目标' in prompt_text
    assert '而不是把用户补充信息当作新的独立闲聊' in prompt_text
    assert '主要交付部分' in prompt_text
    assert '不要只回答其中一个子问题后把必要交付推迟到下一轮' in prompt_text
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('给出完整的最佳努力结果、证据限制和必要步骤',
         '最佳努力'),
        'best-effort result with evidence limitations',
    )
    assert '非必需偏好或排序口径' in prompt_text

    # -- resolver continuation principles --
    assert '解析器续轮原则' in prompt_text
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('返回一个 `speak` action_request', '可见表面动作'),
        'fallback to visible surface action',
    )
    assert '最小缺口' in prompt_text
    assert '缺少可选范围、标准或排序口径不等于缺少必须由用户提供的信息' in prompt_text

    # -- evidence-before-speak rules (generic, not hardcoded capability names) --
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('不要为了给一般判断背书而启动 `web_evidence`',
         '不要为了给一般判断背书而启动',
         '证据能力'),
        'no evidence capability for backing general judgment',
    )
    assert '可行动标准和最后核实步骤' in prompt_text
    assert '只给阻塞说明、可行动标准和最后核实步骤' in prompt_text
    assert '不要继续换同义词重复搜索' in prompt_text
    assert '已有证据和一般判断完成的分析、决策、方案或排查任务' in prompt_text
    assert '分析、决策、方案设计、风险清单或下一步行动' in prompt_text

    # -- internal goal convergence --
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('需要先收束目标、整理优先级、拆解私有后续判断或形成下一步内部目标',
         '内部目标收束'),
        'internal goal convergence concept',
    )
    assert '私有目标收束已经完成' in prompt_text
    assert '没有新的具体私有动作就返回空数组' in prompt_text

    # -- resolver_goal_progress structure --
    assert 'resolver_goal_progress' in prompt_text
    assert '语义进度表' in prompt_text
    assert '不得替换原始目标' in prompt_text
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('`pending`、`partial`、`satisfied`、`blocked`',
         'pending、partial、satisfied、blocked'),
        'deliverable status enum vocabulary',
    )
    _assert_contains_any(
        'ACTION_ROUTER_PROMPT',
        prompt_text,
        ('`resolver_goal_progress.final_response_requirements` 是 L3/dialog 的交付清单',
         'final_response_requirements', '交付清单'),
        'final response requirements as deliverable checklist',
    )
    assert '"final_response_requirements"' in prompt_text


def test_l3_content_plan_scope_preserves_complete_plan_deliverables() -> None:
    """Content plan should leave enough room for multi-part answers."""

    prompt_text = l3_module._CONTENT_PLAN_AGENT_PROMPT

    for required_text in (
        '你决定“本轮要说什么”',
        '`semantic_content` 是本轮用户可见回复的语义载荷',
        '下游只能改写它，不能替你补事实、话题、问题、结论、代码、例子或下一步',
        '不要把生成任务写进 `semantic_content`',
        '不要写“给出一个俏皮回应”',
        '要写“被对方逗乐了，有点小得意',
        '具体要问的内容',
        '固定格式块必须作为一个字符串原样进入 `semantic_content`',
        '缩进、空行、符号和顺序不变',
        '只能把它们整理成本轮内容计划',
        '`selected_text_surface_intent` 是已选择文本输出时传下来的语义目标',
        '已清洗的 resolver observation 摘要',
        '可用于保留已查到的事实、风险和失败边界',
        '合并交付范围',
        '含原始目标、目标进度、deliverables、blockers 或 final_response_requirements',
        '把这些转成可见回答骨架',
        '当前输入可能只是补充约束，不能缩小原始目标',
        '实时、易变或来源绑定的事实',
        '无法确认的部分、可说的泛化范围、核实办法或行动骨架',
        '不要补造具体当前对象、状态、时间或可用性',
        '写出 resolved semantic content',
        '下游不需要再决定“说什么”',
        '`visible_goal` 写交互目的',
        '`voice` 写温度和分寸',
        '`rendering` 写单气泡内的布局、长短和固定格式保护',
        '不能新增事实或话题',
        '轻松接梗',
        '技术对比',
        '保留所有已给数值、单位和结论',
        '固定格式代码',
        '直接包含 fenced code block',
        '证据不足',
        '当前来源未确认 X',
        '本轮可见回复的实际语义载荷；下游可改写但不需要再决定说什么',
        '必须已经写在计划值里',
    ):
        assert required_text in prompt_text

    assert '[DECISION]' not in prompt_text
    assert '[ANSWER]' not in prompt_text
    assert '[SCOPE]' not in prompt_text


def test_l3_style_prompt_does_not_seed_literal_texture_examples() -> None:
    """Style prompt should not turn texture guidance into phrase inventory."""

    prompt_text = l3_module._STYLE_AGENT_PROMPT

    for forbidden_text in (
        '**示例：**',
        '非要我说出来',
        '胸口好像',
        '嗯，我,其实想说',
        '固定感官化比喻',
        '「',
        '」',
    ):
        assert forbidden_text not in prompt_text

    for required_text in (
        '# 工作边界',
        '# 核心转换',
        '# 角色表达依据',
        '## 性格底色怎样影响风格',
        '## 声纹质感怎样影响风格',
        '先处理说话视角风险，再应用角色表达依据',
        '是否没有把声纹描述复制成台词',
        '文本内容计划由独立阶段生成',
        '是否没有让十个声纹维度全部堆进 `linguistic_style`',
    ):
        assert required_text in prompt_text


def test_l3_style_prompt_omits_input_schema_but_explains_fields() -> None:
    """Style prompt should explain consumed fields without a JSON input block."""

    prompt_text = l3_module._STYLE_AGENT_PROMPT

    style_prompt_end = prompt_text.index('# 输出格式')
    style_prompt_body = prompt_text[:style_prompt_end]

    assert '# 输入格式' not in prompt_text
    for required_text in (
        '# 本轮输入字段说明',
        '`logical_stance` 是立场边界',
        '`character_intent` 是行动意图',
        '`internal_monologue` 是上游意识层',
        '`character_mood` 是当前瞬间情绪',
        '`global_vibe` 是当前环境氛围',
        '`last_relationship_insight` 是与当前用户的关系动态',
        '`media_observations` 是当前图片或音频的结构化观察',
        '`interaction_style_context.user_style` 是用户互动风格建议',
        '`group_channel_style` 只在群聊输入中出现',
        '`chat_history` 是最多两条近期表面文本',
        '`decontexualized_input`',
        '`reflection_artifact`',
        '`internal_thought_residue`',
        '# 生成流程',
        '# 输出前自检',
    ):
        assert required_text in style_prompt_body


def test_l3_preference_prompt_stays_out_of_style_ownership() -> None:
    """Preference adapter should not duplicate style-agent responsibility."""

    prompt_text = l3_module._PREFERENCE_ADAPTER_PROMPT

    assert '# 输入格式' not in prompt_text
    for required_text in (
        '# 工作边界',
        '`accepted_user_preferences` 只是下游表达的软约束',
        '不决定一般风格、修辞策略、角色声纹、句长、情绪露出或社交包装',
        '这些属于语言风格阶段和最终台词生成阶段',
        '`linguistic_style` 是上游语言风格约束',
        '只用于检查冲突，不要改写或扩展它',
        '是否没有输出一般风格、角色声纹、修辞策略、内容计划、事实或承诺',
        '# 输出前自检',
        '# 输出格式',
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

    l2d_prompt = ACTION_ROUTER_PROMPT
    for affordance_concept in (
        'action_affordances',
        'resolver_affordances',
        'action_requests',
    ):
        assert affordance_concept in l2d_prompt


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
