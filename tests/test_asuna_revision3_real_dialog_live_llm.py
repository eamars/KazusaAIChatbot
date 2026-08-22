"""Verify that the active Asuna identity reaches real dialog generation."""

from dataclasses import replace
import sys
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.cognition_shared.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import close_db, get_character_profile
from kazusa_ai_chatbot.db.character_identity_growth import (
    get_current_identity,
)
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from tests.llm_trace import write_llm_trace
from tests.test_group_style_output_shape_live_llm import (
    _CapturingLiveLLM,
    _dialog_metrics,
    _dialog_state,
    _skip_if_live_routes_unavailable,
    _surface_input,
)


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_llm,
]


async def test_asuna_revision3_reaches_real_dialog_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run real surface and dialog LLMs with the latest DB identity."""

    await _skip_if_live_routes_unavailable()
    trace_steps: list[dict[str, Any]] = []

    async def capture_trace_step(**kwargs: Any) -> None:
        """Capture protected-stage metadata without writing to production DB."""

        trace_steps.append({
            'stage_name': kwargs.get('stage_name'),
            'route_name': kwargs.get('route_name'),
            'model_name': kwargs.get('model_name'),
            'response_text': kwargs.get('response_text'),
            'parsed_output': kwargs.get('parsed_output'),
            'parse_status': kwargs.get('parse_status'),
            'status': kwargs.get('status'),
            'sequence': kwargs.get('sequence'),
        })

    monkeypatch.setattr(
        llm_tracing,
        'record_llm_trace_step',
        capture_trace_step,
    )
    for recorder_name in (
        'record_llm_stage_event',
        'record_model_contract_event',
        'record_dialog_quality_event',
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )

    try:
        current_identity = await get_current_identity(
            character_id=CHARACTER_GLOBAL_USER_ID,
        )
        profile = await get_character_profile(
            character_id=CHARACTER_GLOBAL_USER_ID,
        )
        assert current_identity['revision_number'] == 3
        assert profile['linguistic_texture_profile']['fragmentation'] == 0.3
        assert profile['linguistic_texture_profile']['rhythmic_bounce'] == 0.5
        assert '完整句' in profile['personality_brief']['tempo']

        case_id = f'revision3-{uuid4().hex}'
        user_input = '我今天有点累，但还想听听你的看法。你会怎么把这件事讲清楚？'
        surface_input = _surface_input(
            case_id=case_id,
            user_input=user_input,
            platform_channel_id=f'asuna-revision3-{uuid4().hex}',
            interaction_style_context='当前没有额外互动风格覆盖。',
            profile=profile,
        )
        expression_context = surface_input['character_expression_context']
        assert '你有轻微碎片感' in expression_context['linguistic_texture']

        text_services = l3_module._build_text_surface_services()
        surface_llm = _CapturingLiveLLM('surface', text_services.llm)
        text_services = replace(text_services, llm=surface_llm)
        surface_output = await run_text_surface_planning(
            surface_input,
            text_services,
        )

        generator_llm = _CapturingLiveLLM(
            'dialog_generator',
            dialog_module._dialog_generator_llm,
        )
        monkeypatch.setattr(
            dialog_module,
            '_dialog_generator_llm',
            generator_llm,
        )
        dialog_output = await dialog_module.dialog_generator(
            _dialog_state(
                surface_input=surface_input,
                surface_output=surface_output,
                profile=profile,
                style_arm='runtime_revision3',
                case_id=case_id,
            )
        )
        final_dialog = dialog_output['final_dialog']
        evidence = {
            'run_context': {
                'database_profile_source': 'runtime MongoDB identity ledger',
                'character_id': CHARACTER_GLOBAL_USER_ID,
                'revision_number': current_identity['revision_number'],
                'database_writes': 'trace and event writers patched to memory',
                'real_llm_calls': True,
            },
            'input': {
                'user_message': user_input,
                'surface_input': surface_input,
            },
            'profile_fields_consumed': {
                'fragmentation': profile[
                    'linguistic_texture_profile'
                ]['fragmentation'],
                'rhythmic_bounce': profile[
                    'linguistic_texture_profile'
                ]['rhythmic_bounce'],
                'tempo': profile['personality_brief']['tempo'],
            },
            'surface_model_calls': surface_llm.calls,
            'text_surface_output': surface_output,
            'dialog_generator_calls': generator_llm.calls,
            'dialog_output': dialog_output,
            'final_dialog_metrics': _dialog_metrics(final_dialog),
            'trace_steps': trace_steps,
            'verification': {
                'profile_revision_reached_test': True,
                'profile_descriptor_reached_surface_input': True,
                'surface_llm_called': bool(surface_llm.calls),
                'dialog_generator_called': bool(generator_llm.calls),
                'final_dialog_non_empty': bool(final_dialog),
            },
        }
        trace_path = write_llm_trace(
            'asuna_revision3_real_dialog_live_llm',
            'runtime_profile_single_case',
            evidence,
        )

        assert trace_path.exists()
        assert surface_llm.calls
        assert generator_llm.calls
        assert isinstance(final_dialog, list) and final_dialog
    finally:
        await close_db()
