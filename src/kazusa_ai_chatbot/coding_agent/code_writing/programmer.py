"""Programmer worker for module-level code-writing contracts."""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PROGRAMMER_TARGET_INPUT_TOKEN_CAP,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ModuleProgrammerContract,
    ModuleProgrammerResult,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PROGRAMMER_LLM_API_KEY,
    CODING_AGENT_PROGRAMMER_LLM_BASE_URL,
    CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PROGRAMMER_LLM_MODEL,
    CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)

PROGRAMMER_LLM_CALL_TIMEOUT_SECONDS = 300


MODULE_PROGRAMMER_PROMPT = '''\
You are a programmer for one file or one module section.
You receive one module contract. Return the requested file content only.

# Contract Fields
- file_label: a label for the work. It is not a path.
- edit_mode:
  - complete_file: write the whole file.
  - symbol_bundle: write only the requested imports and top-level symbols that
    should be inserted or replaced.
- content_format:
  - python: return Python source.
  - text: return plain text or Markdown source.
- imports: required import lines that must appear in this output.
- current_file_context: short description of existing names in the file.
- symbols_to_define: required top-level module names and signatures for
  Python, or required text sections for text.
- required_behavior: behavior checks the code must satisfy.

# Rules
- Return one markdown fenced block and nothing else.
- For content_format python, use a python code block and include only Python
  source inside the block.
- For content_format text, use a markdown or text code block and include only
  the requested text content inside the block.
- For complete_file, include the complete requested file content.
- For symbol_bundle, include only the requested imports and requested symbols
  or text sections. Do not rewrite the whole file.
- You may add supplementary imports from the Python standard library when they
  are needed by requested Python code.
- For Python, use the names and signatures from symbols_to_define exactly.
- For Python, implement every listed symbol with real executable Python.
- For Python, do not copy explanatory phrases from body_contract into code
  comments when the phrase is not valid implementation detail.
- For text, write the named sections requested by symbols_to_define.
- Do not use pass, ellipsis, NotImplementedError, TODO, placeholder comments,
  JSON, diffs, or file path comments.
- Treat current_file_context as the list of existing code names available in
  the target file.
- Call an existing helper only when its exact name appears in
  current_file_context, imports, or symbols_to_define.
- Do not invent file paths, patch anchors, peer outputs, or command execution.
- For Python, you may add private helper functions or private variables only
  when they are needed by the requested symbols.
- For Python, do not add project imports or third-party imports unless they
  are listed in imports.

# Output Format
Return only one markdown fenced block. The block must contain the requested
source content.
'''

MODULE_PROGRAMMER_RETRY_PROMPT = '''\
Your previous response did not match the required output format.
Return only one markdown fenced block containing the requested source content.
Do not return JSON, prose outside the block, patch text, file path comments, or
multiple code blocks.
'''

_module_programmer_llm = LLInterface()
_module_programmer_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.module_programmer",
    route_name="CODING_AGENT_PROGRAMMER_LLM",
    base_url=CODING_AGENT_PROGRAMMER_LLM_BASE_URL,
    api_key=CODING_AGENT_PROGRAMMER_LLM_API_KEY,
    model=CODING_AGENT_PROGRAMMER_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
    ),
)


async def run_module_programmer_contract(
    *,
    module_contract: ModuleProgrammerContract,
    trace: dict[str, object] | None = None,
) -> ModuleProgrammerResult:
    """Run one module-level programmer contract and return source content."""

    payload_text = json.dumps(module_contract, ensure_ascii=False, indent=2)
    context_budget = prompt_budget_metadata(
        system_prompt=MODULE_PROGRAMMER_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PROGRAMMER_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=[],
    )
    if context_budget["over_hard_cap"]:
        result: ModuleProgrammerResult = {
            "code_artifact": "",
        }
        _fill_module_trace(
            trace,
            raw_output="",
            result=result,
            context_budget=context_budget,
            blocked_before_invoke=True,
            markdown_code_block_found=False,
        )
        return result

    raw_output, attempts = await _invoke_module_programmer(payload_text)
    content_format = _content_format(module_contract)
    code_artifact = _extract_single_output_block(
        raw_output,
        content_format=content_format,
    )
    result = {
        "code_artifact": code_artifact,
    }
    _fill_module_trace(
        trace,
        raw_output=raw_output,
        result=result,
        context_budget=context_budget,
        blocked_before_invoke=False,
        markdown_code_block_found=bool(code_artifact),
        attempts=attempts,
    )
    return result


async def _invoke_module_programmer(
    payload_text: str,
) -> tuple[str, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    raw_output = ""
    for attempt_index in range(2):
        messages = [
            SystemMessage(content=MODULE_PROGRAMMER_PROMPT),
            HumanMessage(content=payload_text),
        ]
        if attempt_index > 0:
            messages.append(HumanMessage(content=MODULE_PROGRAMMER_RETRY_PROMPT))
        timed_out = False
        try:
            response = await asyncio.wait_for(
                _module_programmer_llm.ainvoke(
                    messages,
                    config=_module_programmer_llm_config,
                ),
                timeout=PROGRAMMER_LLM_CALL_TIMEOUT_SECONDS,
            )
            raw_output = response.content
        except asyncio.TimeoutError:
            timed_out = True
            raw_output = ""
        code_artifact = _extract_single_output_block(
            raw_output,
            content_format=_content_format_from_payload(payload_text),
        )
        attempts.append({
            "attempt": attempt_index + 1,
            "raw_output": raw_output,
            "markdown_code_block_found": bool(code_artifact),
            "timed_out": timed_out,
        })
        if code_artifact or timed_out:
            break
    return raw_output, attempts


def _extract_single_output_block(
    raw_output: str,
    *,
    content_format: str,
) -> str:
    if not isinstance(raw_output, str):
        return ""

    matches = re.findall(
        r"```([A-Za-z0-9_-]*)\s*\n(.*?)\n```",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(matches) != 1:
        return ""

    language, content = matches[0]
    normalized_language = language.strip().casefold()
    if content_format == "python" and normalized_language not in {"python", "py"}:
        return ""
    if content_format == "text" and normalized_language not in {
        "markdown",
        "md",
        "text",
        "txt",
        "",
    }:
        return ""
    code_artifact = content.strip()
    return code_artifact


def _content_format(module_contract: ModuleProgrammerContract) -> str:
    content_format = module_contract.get("content_format", "python")
    if content_format == "text":
        return "text"
    return "python"


def _content_format_from_payload(payload_text: str) -> str:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return "python"
    if isinstance(payload, dict) and payload.get("content_format") == "text":
        return "text"
    return "python"


def _fill_module_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    result: ModuleProgrammerResult,
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
    markdown_code_block_found: bool,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _module_programmer_llm_config.route_name
    trace["model"] = _module_programmer_llm_config.model
    trace["thinking_enabled"] = CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["normalized_output"] = result
    trace["markdown_code_block_found"] = markdown_code_block_found
    if attempts is not None:
        trace["attempts"] = attempts
