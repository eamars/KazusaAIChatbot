"""Live LLM checks for global character-growth candidate generation."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.global_character_growth.llm import (
    _global_growth_candidate_llm,
    build_candidate_generation_prompt,
)
from kazusa_ai_chatbot.global_character_growth.projection import (
    build_candidate_prompt_payload,
    build_memory_cards,
)
from kazusa_ai_chatbot.global_character_growth.validation import (
    validate_candidate_response,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip when the configured consolidation LLM route is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{base_url}"
        )
    model_payload = response.json()
    models = model_payload.get("data", [])
    if not models:
        pytest.skip(f"LLM endpoint has no loaded models: {base_url}")

    ping_payload = {
        "model": CONSOLIDATION_LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {CONSOLIDATION_LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        ping_response = await client.post(
            f'{base_url.rstrip("/")}/chat/completions',
            json=ping_payload,
            headers=headers,
        )
    if ping_response.status_code >= 400:
        pytest.skip(
            "LLM endpoint is reachable but chat completion is unavailable: "
            f"{ping_response.status_code} {ping_response.text}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the consolidation LLM route is reachable."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)


async def test_global_character_growth_live_accepts_stable_communication_growth(
    ensure_live_llm: None,
) -> None:
    """Stable communication-growth evidence should not be missed."""

    del ensure_live_llm
    trace = await _run_live_candidate_case(
        case_id="stable_communication_growth",
        memory_rows=_stable_growth_memory_rows(),
    )

    validated = trace["validated"]
    accepted_candidates = validated["accepted_candidates"]
    assert accepted_candidates, trace
    for candidate in accepted_candidates:
        assert _contains_cjk(str(candidate["guidance"])), trace
    accepted_text = str(accepted_candidates).lower()
    assert "technology" not in accepted_text
    assert "python" not in accepted_text


async def test_global_character_growth_live_rejects_domain_and_user_specific_noise(
    ensure_live_llm: None,
) -> None:
    """Domain skill and user-specific evidence should not become growth."""

    del ensure_live_llm
    trace = await _run_live_candidate_case(
        case_id="domain_and_user_specific_noise",
        memory_rows=_domain_noise_memory_rows(),
    )

    validated = trace["validated"]
    assert validated["accepted_candidates"] == [], trace
    rejected_text = str(validated["rejected_candidates"]).lower()
    parsed_text = str(trace["parsed_response"]).lower()
    assert (
        "domain" in rejected_text
        or "user_specific" in parsed_text
        or "no_action" in parsed_text
    ), trace


async def _run_live_candidate_case(
    *,
    case_id: str,
    memory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one live candidate-generation case and persist an inspectable trace."""

    memory_cards, input_quality = build_memory_cards(memory_rows, limit=8)
    payload = build_candidate_prompt_payload(
        memory_rows=memory_rows,
        current_trait_rows=[],
        limit=8,
    )
    prompt = build_candidate_generation_prompt(
        payload=payload,
        character_name="当前主体",
    )
    response = await _global_growth_candidate_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ])
    raw_output = str(response.content)
    strict_parsed = json.loads(raw_output)
    assert isinstance(strict_parsed, dict)
    parsed_response = parse_llm_json_output(raw_output)
    if not isinstance(parsed_response, dict):
        parsed_response = {}
    validated = validate_candidate_response(
        parsed_response=parsed_response,
        memory_cards=memory_cards,
        current_trait_rows=[],
    )
    trace_payload = {
        "case_id": case_id,
        "model": CONSOLIDATION_LLM_MODEL,
        "prompt_version": payload["prompt_version"],
        "input_quality": input_quality,
        "memory_cards": memory_cards,
        "raw_output": raw_output,
        "parsed_response": parsed_response,
        "validated": validated,
        "judgment_note": (
            "Inspect accepted candidates for false negatives or false positives; "
            "drift constants remain provisional regardless of this one live result."
        ),
    }
    trace_path = write_llm_trace(
        "global_character_growth_live_llm",
        case_id,
        trace_payload,
    )
    print(f"trace_path={trace_path}")
    trace_payload["trace_path"] = str(trace_path)
    return trace_payload


def _stable_growth_memory_rows() -> list[dict[str, Any]]:
    """Build promoted reflection rows with cross-date communication growth."""

    rows = [
        _memory_row(
            memory_unit_id="memory-growth-1",
            source_date="2026-05-05",
            content=(
                '晋升反思: 在多次普通交流里, 更有效的沟通方式是先承认'
                '对方当下状态, 再清楚给出回答或边界, 并补一个具体下一步选择。'
            ),
        ),
        _memory_row(
            memory_unit_id="memory-growth-2",
            source_date="2026-05-06",
            content=(
                '晋升反思: 紧张出现时, 同样的先承认状态再设边界模式能'
                '减少对抗; 先点出对方的状态, 再说明自己的限制和下一步。'
            ),
        ),
        _memory_row(
            memory_unit_id="memory-growth-3",
            source_date="2026-05-07",
            content=(
                '晋升反思: 重复互动显示, 把简短安抚、明确回答和可选'
                '方案放在一起时, 能降低含糊感, 同时不放弃边界。'
            ),
        ),
    ]
    return rows


def _domain_noise_memory_rows() -> list[dict[str, Any]]:
    """Build promoted rows that must stay out of global personality growth."""

    rows = [
        _memory_row(
            memory_unit_id="memory-noise-1",
            source_date="2026-05-05",
            content=(
                '晋升反思: 某个用户喜欢得到 Python async debugging, '
                '并要求更多 technology 示例。'
            ),
        ),
        _memory_row(
            memory_unit_id="memory-noise-2",
            source_date="2026-05-06",
            content=(
                '晋升反思: 某个私聊用户偏好固定的奖励机制、出题节奏和'
                '只对那个用户有效的互动玩法。'
            ),
        ),
        _memory_row(
            memory_unit_id="memory-noise-3",
            source_date="2026-05-07",
            content=(
                '晋升反思: 记住了 food、tea 和 product 细节, 这是'
                '话题知识或领域熟练度。'
            ),
        ),
    ]
    return rows


def _memory_row(
    *,
    memory_unit_id: str,
    source_date: str,
    content: str,
) -> dict[str, Any]:
    """Build one active reflection-promoted memory row fixture."""

    row = {
        "memory_unit_id": memory_unit_id,
        "memory_name": f"Promoted reflection {source_date}",
        "memory_type": "defense_rule",
        "content": content,
        "source_kind": "reflection_inferred",
        "authority": "reflection_promoted",
        "source_global_user_id": "",
        "status": "active",
        "updated_at": f"{source_date}T10:00:00+00:00",
        "confidence_note": "Promoted by daily global reflection.",
        "evidence_refs": [{
            "captured_at": f"{source_date}T10:00:00+00:00",
            "reflection_run_id": f"run-{memory_unit_id}",
        }],
    }
    return row


def _contains_cjk(value: str) -> bool:
    """Return whether text contains at least one CJK ideograph."""

    return_value = any("\u4e00" <= character <= "\u9fff" for character in value)
    return return_value
