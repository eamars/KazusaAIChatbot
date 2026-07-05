import json
import os
from pathlib import Path

import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching import source_intake
from kazusa_ai_chatbot.coding_agent.code_fetching import source_resolver
from tests.llm_trace import write_llm_trace


_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / 'fixtures'
    / 'coding_agent_source_intake_signoff_cases.json'
)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding='utf-8'))


def test_source_intake_signoff_fixture_contract() -> None:
    fixture = _load_fixture()
    cases = fixture['cases']
    case_ids = [case['case_id'] for case in cases]
    statuses = {case['expected']['public_status'] for case in cases}

    assert fixture['schema_version'] == (
        'coding_agent_source_intake_signoff_cases.v1'
    )
    assert fixture['execution_contract']['real_llm_intake_required'] is True
    assert fixture['execution_contract']['run_cases_one_at_a_time'] is True
    assert len(cases) == 20
    assert len(set(case_ids)) == 20
    assert statuses == {'succeeded', 'rejected', 'needs_user_input'}

    for case in cases:
        assert case['task_text'].strip()
        assert 'expected' in case
        assert 'public_status' in case['expected']
        assert 'task_source_mode' in case['expected']
        assert 'forbidden_failure_modes' in case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_source_intake_signoff_case_live_llm() -> None:
    case_id = os.environ.get('CODING_AGENT_SOURCE_INTAKE_SIGNOFF_CASE_ID')
    if not case_id:
        pytest.skip('Set CODING_AGENT_SOURCE_INTAKE_SIGNOFF_CASE_ID.')

    fixture = _load_fixture()
    cases = {case['case_id']: case for case in fixture['cases']}
    if case_id not in cases:
        pytest.fail(f'Unknown signoff case id: {case_id}')

    case = cases[case_id]
    expected = case['expected']
    request = dict(case.get('request_fields', {}))
    request['question'] = case['task_text']

    intake_result = await source_intake.run_source_intake(case['task_text'])
    resolution = source_resolver.resolve_source_request(request, intake_result)

    trace_path = write_llm_trace(
        'coding_agent_source_intake_signoff',
        case_id,
        {
            'task_text': case['task_text'],
            'request_fields': case.get('request_fields', {}),
            'intake_result': source_intake.source_intake_result_to_dict(
                intake_result
            ),
            'resolution': source_resolver.source_resolution_to_dict(
                resolution
            ),
            'expected': expected,
        },
    )

    assert trace_path.exists()
    assert resolution.status == expected['public_status']
    assert intake_result.task_source_mode == expected['task_source_mode']
    assert resolution.issue_code == expected['issue_code']

    primary = expected.get('primary')
    if primary is None:
        assert resolution.source is None
        return

    assert resolution.source is not None
    assert resolution.source.owner == primary['owner']
    assert resolution.source.repo == primary['repo']
    assert resolution.source.source_kind == primary['scope_kind']
    assert resolution.source.repo_relative_path == primary['repo_relative_path']
    if 'requested_ref' in primary:
        assert resolution.source.requested_ref == primary['requested_ref']
