"""LLM-backed file/module product-manager decisions for code writing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PM_TARGET_INPUT_TOKEN_CAP,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ModulePMInput,
    ModuleProgrammerContentFormat,
    ModuleProgrammerContract,
    ModuleProgrammerEditMode,
    ModuleProgrammerSymbol,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PM_LLM_API_KEY,
    CODING_AGENT_PM_LLM_BASE_URL,
    CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PM_LLM_MODEL,
    CODING_AGENT_PM_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

MAX_TEXT_FIELD_CHARS = 700
MAX_QUESTION_CHARS = 4000
MAX_SOURCE_CONTEXT_ROWS = 8
MAX_SOURCE_CONTEXT_TEXT_CHARS = 1600
MAX_LIST_ITEMS = 10
SUB_PM_LLM_CALL_TIMEOUT_SECONDS = 300
MODULE_PROGRAMMER_EDIT_MODES: tuple[ModuleProgrammerEditMode, ...] = (
    "complete_file",
    "symbol_bundle",
)
MODULE_PROGRAMMER_CONTENT_FORMATS: tuple[ModuleProgrammerContentFormat, ...] = (
    "python",
    "text",
)


MODULE_PM_CONTRACT_PROMPT = '''\
You are a Module PM for a code-writing agent.
You receive one accepted file assignment. Create one programmer contract for
that file or module. The programmer will use only your contract.

# Contract Fields
- file_label: stable label for this file assignment. It is not a path.
- edit_mode:
  - complete_file: programmer writes a whole file.
  - symbol_bundle: programmer writes imports plus requested top-level symbols.
- content_format:
  - python: programmer writes Python source.
  - text: programmer writes plain text or Markdown source.
- module_purpose: why this file or module exists.
- lifecycle_owner: the declared runtime owner that holds state for this module.
- provided_interfaces: public interfaces this module provides.
- consumed_interfaces: interfaces this module consumes from other modules.
- existing_source_anchors: existing classes, functions, or sections that must
  be preserved or extended.
- integration_behaviors: how this module fits the wider feature.
- cross_slice_interfaces: compact summaries of interfaces from other module
  slices that this module consumes.
- imports: required import lines. Pass these through when they are needed.
- current_file_context: existing names and behavior available in this file.
  This is a bounded excerpt. It may be shorter than the full source file.
- source_file_chars: total character count of the original source file.
  0 for new files.
- selected_evidence: source facts relevant to this file assignment.
- required_behavior: observable behavior this file assignment must satisfy.

# Rules
- Return one JSON object only.
- Create one module-level programmer contract.
- Preserve content_format from the input.
- Use imports as the only external-name channel.
- Do not invent project imports. Use only import lines supplied in the input
  or exact imported names visible in current_file_context or selected_evidence.
- Use exact Python identifiers visible in current_file_context,
  selected_evidence, provided_interfaces, existing_source_anchors, or imports.
  Do not rename existing classes, functions, methods, or helpers.
- Use symbols_to_define to list new public or test symbols the programmer must
  write for Python, or the new named sections the programmer must write for text.
- Use symbols_to_modify to list existing classes, functions, methods, or
  sections the programmer must extend or replace. Every entry must reference a
  name visible in current_file_context or existing_source_anchors.
- If source_file_chars is larger than the current_file_context length, the
  source file was too large to include in full. Only reference symbols visible
  in current_file_context or existing_source_anchors. Do not guess at symbols
  that might exist beyond the bounded excerpt.
- Give each symbol or section a name, kind, signature, and body_contract.
- Put required fields or methods for a class in children.
- For Python signatures, write valid Python declarations or assignments only.
- Keep current_file_context bounded to the context you received.
- Do not write Python bodies, patch hunks, file paths, command steps, or peer
  programmer output.

# Output Format
{
  "file_label": "same file_label as input",
  "edit_mode": "complete_file | symbol_bundle",
  "content_format": "python | text",
  "module_purpose": "plain-language purpose for this file or module",
  "lifecycle_owner": "declared runtime owner that holds state",
  "provided_interfaces": [{"name": "...", "kind": "...", "contract": "..."}],
  "consumed_interfaces": [{"name": "...", "provider_slice_id": "...", "contract": "..."}],
  "existing_source_anchors": [{"name": "...", "kind": "...", "required_action": "preserve | extend | call | replace"}],
  "imports": ["required import line"],
  "current_file_context": "bounded context string",
  "symbols_to_define": [
    {
      "name": "new symbol name",
      "kind": "module_variable | dataclass | class | function | method | test | section",
      "signature": "exact declaration, assignment shape, or section heading",
      "body_contract": "plain-language behavior or text the item must implement",
      "children": []
    }
  ],
  "symbols_to_modify": [
    {
      "name": "existing symbol name from current_file_context or existing_source_anchors",
      "kind": "class | function | method | section",
      "signature": "exact current declaration",
      "body_contract": "plain-language change to apply while preserving existing lifecycle",
      "children": []
    }
  ],
  "required_behavior": ["observable behavior"]
}
'''

MODULE_PM_CONTRACT_RETRY_PROMPT = '''\
Your previous Module PM response was empty or not valid JSON.
Return one strict JSON object only, matching the required output format from
the system instructions. Do not include markdown, commentary, or code fences.
'''

_module_pm_module_llm = LLInterface()
_module_pm_module_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.module_pm_module_contract",
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def decide_module_programmer_contract(
    module_pm_input: ModulePMInput,
    *,
    trace: dict[str, object] | None = None,
) -> ModuleProgrammerContract:
    """Ask the Module PM for one module-level programmer contract."""

    payload = _module_pm_module_payload(module_pm_input)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    context_budget = prompt_budget_metadata(
        system_prompt=MODULE_PM_CONTRACT_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PM_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
    )
    if context_budget["over_hard_cap"]:
        contract = _empty_module_programmer_contract(
            module_pm_input=module_pm_input,
            reason="Module PM prompt exceeded the context budget.",
        )
        _fill_module_contract_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_output=contract,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return contract

    raw_output, parsed, attempts = await _invoke_module_pm_module_json(
        payload_text,
    )
    contract = normalize_module_programmer_contract(
        parsed,
        module_pm_input=module_pm_input,
    )
    _fill_module_contract_trace(
        trace,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_output=contract,
        context_budget=context_budget,
        blocked_before_invoke=False,
        attempts=attempts,
    )
    return contract


async def _invoke_module_pm_module_json(
    payload_text: str,
) -> tuple[str, object, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    raw_output = ""
    parsed: object = {}
    for attempt_index in range(2):
        messages = [
            SystemMessage(content=MODULE_PM_CONTRACT_PROMPT),
            HumanMessage(content=payload_text),
        ]
        if attempt_index > 0:
            messages.append(HumanMessage(content=MODULE_PM_CONTRACT_RETRY_PROMPT))
        timed_out = False
        try:
            response = await asyncio.wait_for(
                _module_pm_module_llm.ainvoke(
                    messages,
                    config=_module_pm_module_llm_config,
                ),
                timeout=SUB_PM_LLM_CALL_TIMEOUT_SECONDS,
            )
            raw_output = response.content
            parsed = parse_llm_json_output(raw_output)
        except asyncio.TimeoutError:
            timed_out = True
            raw_output = ""
            parsed = {}
        attempts.append({
            "attempt": attempt_index + 1,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "timed_out": timed_out,
        })
        if parsed or timed_out:
            break
    return raw_output, parsed, attempts


def normalize_module_programmer_contract(
    parsed: object,
    *,
    module_pm_input: ModulePMInput,
) -> ModuleProgrammerContract:
    """Normalize Module PM JSON into the module programmer contract."""

    if not isinstance(parsed, dict):
        contract = _empty_module_programmer_contract(
            module_pm_input=module_pm_input,
            reason="Module PM returned malformed output.",
        )
        return contract

    edit_mode = _bounded_text(parsed.get("edit_mode"))
    if edit_mode not in MODULE_PROGRAMMER_EDIT_MODES:
        edit_mode = module_pm_input["edit_mode"]

    content_format = _bounded_text(parsed.get("content_format"))
    if content_format not in MODULE_PROGRAMMER_CONTENT_FORMATS:
        content_format = module_pm_input["content_format"]

    imports = _string_list(parsed.get("imports"), MAX_LIST_ITEMS)
    if not imports:
        imports = _string_list(module_pm_input["imports"], MAX_LIST_ITEMS)

    current_file_context = _bounded_multiline_text(
        parsed.get("current_file_context"),
        MAX_SOURCE_CONTEXT_TEXT_CHARS,
    )
    if not current_file_context:
        current_file_context = _bounded_multiline_text(
            module_pm_input["current_file_context"],
            MAX_SOURCE_CONTEXT_TEXT_CHARS,
        )

    required_behavior = _string_list(
        parsed.get("required_behavior"),
        MAX_LIST_ITEMS,
    )
    if not required_behavior:
        required_behavior = _string_list(
            module_pm_input["required_behavior"],
            MAX_LIST_ITEMS,
        )

    contract: ModuleProgrammerContract = {
        "file_label": (
            _bounded_text(parsed.get("file_label"))
            or module_pm_input["file_label"]
        ),
        "edit_mode": edit_mode,
        "content_format": content_format,
        "module_purpose": (
            _bounded_text(parsed.get("module_purpose"))
            or module_pm_input["module_purpose"]
        ),
        "lifecycle_owner": (
            _bounded_text(parsed.get("lifecycle_owner"))
            or module_pm_input["lifecycle_owner"]
        ),
        "provided_interfaces": _dict_list(
            parsed.get("provided_interfaces"),
            MAX_LIST_ITEMS,
        ) or module_pm_input["provided_interfaces"],
        "consumed_interfaces": _dict_list(
            parsed.get("consumed_interfaces"),
            MAX_LIST_ITEMS,
        ) or module_pm_input["consumed_interfaces"],
        "existing_source_anchors": _dict_list(
            parsed.get("existing_source_anchors"),
            MAX_LIST_ITEMS,
        ) or module_pm_input["existing_source_anchors"],
        "imports": imports,
        "current_file_context": current_file_context,
        "symbols_to_define": _module_symbols_from_parsed(
            parsed.get("symbols_to_define"),
        ),
        "symbols_to_modify": _module_symbols_from_parsed(
            parsed.get("symbols_to_modify"),
        ),
        "required_behavior": required_behavior,
    }
    return contract


def _module_pm_module_payload(
    module_pm_input: ModulePMInput,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "file_label": _bounded_text(module_pm_input["file_label"]),
        "edit_mode": module_pm_input["edit_mode"],
        "content_format": module_pm_input["content_format"],
        "module_purpose": _bounded_multiline_text(
            module_pm_input["module_purpose"],
            MAX_QUESTION_CHARS,
        ),
        "lifecycle_owner": _bounded_text(
            module_pm_input["lifecycle_owner"],
        ),
        "provided_interfaces": _dict_list(
            module_pm_input["provided_interfaces"],
            MAX_LIST_ITEMS,
        ),
        "consumed_interfaces": _dict_list(
            module_pm_input["consumed_interfaces"],
            MAX_LIST_ITEMS,
        ),
        "existing_source_anchors": _dict_list(
            module_pm_input["existing_source_anchors"],
            MAX_LIST_ITEMS,
        ),
        "integration_behaviors": _string_list(
            module_pm_input["integration_behaviors"],
            MAX_LIST_ITEMS,
        ),
        "imports": _string_list(module_pm_input["imports"], MAX_LIST_ITEMS),
        "current_file_context": _bounded_multiline_text(
            module_pm_input["current_file_context"],
            MAX_SOURCE_CONTEXT_TEXT_CHARS,
        ),
        "source_file_chars": module_pm_input.get("source_file_chars", 0),
        "selected_evidence": _compact_source_context(
            module_pm_input["selected_evidence"],
        ),
        "required_behavior": _string_list(
            module_pm_input["required_behavior"],
            MAX_LIST_ITEMS,
        ),
    }
    cross_slice_interfaces = module_pm_input.get("cross_slice_interfaces")
    if cross_slice_interfaces:
        payload["cross_slice_interfaces"] = _dict_list(
            cross_slice_interfaces,
            MAX_LIST_ITEMS,
        )
    module_contract_feedback = module_pm_input.get("module_contract_feedback")
    if module_contract_feedback is not None:
        payload["module_contract_feedback"] = module_contract_feedback
    return payload


def _module_symbols_from_parsed(parsed: object) -> list[ModuleProgrammerSymbol]:
    if not isinstance(parsed, list):
        return []

    symbols: list[ModuleProgrammerSymbol] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        symbol = _module_symbol_from_dict(item)
        if symbol is None:
            continue
        symbols.append(symbol)
        if len(symbols) >= MAX_LIST_ITEMS:
            break
    return symbols


def _module_symbol_from_dict(
    item: dict[str, Any],
) -> ModuleProgrammerSymbol | None:
    name = _bounded_text(item.get("name"))
    signature = _bounded_text(item.get("signature"))
    body_contract = _bounded_text(item.get("body_contract"))
    if not name and not signature and not body_contract:
        return None

    symbol: ModuleProgrammerSymbol = {
        "name": name,
        "kind": _bounded_text(item.get("kind")) or "function",
        "signature": signature,
        "body_contract": body_contract,
        "children": _module_symbol_children(item.get("children")),
    }
    return symbol


def _module_symbol_children(parsed: object) -> list[object]:
    if not isinstance(parsed, list):
        return []

    children: list[object] = []
    for item in parsed:
        if isinstance(item, dict):
            symbol = _module_symbol_from_dict(item)
            if symbol is not None:
                children.append(symbol)
        else:
            text = _bounded_text(item)
            if text:
                children.append(text)
        if len(children) >= MAX_LIST_ITEMS:
            break
    return children


def _empty_module_programmer_contract(
    *,
    module_pm_input: ModulePMInput,
    reason: str,
) -> ModuleProgrammerContract:
    contract: ModuleProgrammerContract = {
        "file_label": module_pm_input["file_label"],
        "edit_mode": module_pm_input["edit_mode"],
        "content_format": module_pm_input["content_format"],
        "module_purpose": module_pm_input["module_purpose"],
        "lifecycle_owner": module_pm_input["lifecycle_owner"],
        "provided_interfaces": module_pm_input["provided_interfaces"],
        "consumed_interfaces": module_pm_input["consumed_interfaces"],
        "existing_source_anchors": module_pm_input["existing_source_anchors"],
        "imports": _string_list(module_pm_input["imports"], MAX_LIST_ITEMS),
        "current_file_context": _bounded_multiline_text(
            module_pm_input["current_file_context"],
            MAX_SOURCE_CONTEXT_TEXT_CHARS,
        ),
        "symbols_to_define": [],
        "symbols_to_modify": [],
        "required_behavior": [reason],
    }
    return contract


def _fill_module_contract_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: object,
    normalized_output: ModuleProgrammerContract,
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _module_pm_module_llm_config.route_name
    trace["model"] = _module_pm_module_llm_config.model
    trace["thinking_enabled"] = CODING_AGENT_PM_LLM_THINKING_ENABLED
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
    if attempts is not None:
        trace["attempts"] = attempts


def _compact_source_context(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    compact_rows: list[dict[str, object]] = []
    for row in rows[:MAX_SOURCE_CONTEXT_ROWS]:
        compact_row = dict(row)
        text = compact_row.get("text")
        if isinstance(text, str):
            compact_row["text"] = _bounded_multiline_text(
                text,
                MAX_SOURCE_CONTEXT_TEXT_CHARS,
            )
        compact_rows.append(compact_row)
    return compact_rows


def _dict_list(value: object, limit: int) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        text = _bounded_text(item)
        if not text or text in strings:
            continue
        strings.append(text)
        if len(strings) >= limit:
            break
    return strings


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    if len(text) > MAX_TEXT_FIELD_CHARS:
        text = text[:MAX_TEXT_FIELD_CHARS].rstrip()
    return text


def _bounded_multiline_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text
