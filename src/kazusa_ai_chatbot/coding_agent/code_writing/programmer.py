"""Programmer worker for one new-artifact contract."""

from __future__ import annotations

import ast
import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PROGRAMMER_TARGET_INPUT_TOKEN_CAP,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    WritingContentFormat,
    WritingProgrammerContract,
    WritingProgrammerResult,
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


WRITING_PROGRAMMER_PROMPT = '''\
You are a programmer writing one new artifact.
You receive one artifact contract. Return the requested artifact content only.

# Contract Fields
- artifact_id: stable id for this artifact.
- file_label: human-readable label. It is not a path.
- file_kind: source, test, docs, config, or data.
- content_format: python, markdown, text, json, or csv.
- purpose: what this artifact must provide.
- imports: required imports that must be satisfied.
- provided_interfaces: interfaces this artifact provides.
- consumed_interfaces: interfaces this artifact may rely on from other
  generated artifacts.
- required_behavior: behavior this artifact must satisfy.

# Rules
- Return one markdown fenced block and nothing else.
- For python, use a python code block and include only Python source.
- For markdown, use an outer tilde fence: ~~~markdown. Keep example code
  blocks inside the Markdown as ordinary triple-backtick blocks.
- For json, csv, or text, use a code block with that language label. For
  config text, a precise label such as toml, yaml, ini, or env is also allowed.
- Write the complete artifact content, not a diff.
- Follow the artifact contract exactly. Do not replace it with a nearby
  generic task, broader project, or different file behavior.
- Include every required import. You may add Python standard library imports
  when needed by the code.
- When consumed_interfaces are listed, use imports or ordinary calls to those
  interfaces. Do not redefine, stub, or fake consumed interfaces inside the
  artifact unless the contract explicitly asks for a mock in test code.
- For test artifacts, write assertions against the consumed interface contract
  and required_behavior. Do not invent a different return shape, argument
  shape, or interface name.
- Do not invent file paths, patch anchors, command output, package installs,
  peer programmer output, or JSON metadata around the artifact.
- Do not leave unfinished placeholder comments, pass-only function/class/module
  bodies, ellipsis-only bodies, NotImplementedError, or prose outside the
  block.
- When handling external uncertainty, such as network or file-system errors,
  return an explicit fallback value or record a clear error value.
- For Python, catch specific exception types only when handling external
  uncertainty. Let ordinary internal bugs surface.

# Output Format
Return only one markdown fenced block containing the requested artifact.
'''

WRITING_PROGRAMMER_RETRY_PROMPT = '''\
Your previous response was not accepted.

Problems:
{diagnostics}

Return only one markdown fenced block containing the requested artifact.
Do not return JSON, prose outside the block, patch text, file path comments, or
multiple code blocks.
For markdown artifacts, wrap the whole artifact in ~~~markdown and close it
with ~~~. Do not add any other tilde fence inside the artifact content.
'''

_writing_programmer_llm = LLInterface()
_writing_programmer_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.writing_programmer",
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


async def run_writing_programmer_contract(
    *,
    artifact_contract: WritingProgrammerContract,
    trace: dict[str, object] | None = None,
) -> WritingProgrammerResult:
    """Run one new-artifact programmer contract and return content."""

    payload_text = json.dumps(artifact_contract, ensure_ascii=False, indent=2)
    context_budget = prompt_budget_metadata(
        system_prompt=WRITING_PROGRAMMER_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PROGRAMMER_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=[],
    )
    if context_budget["over_hard_cap"]:
        result = _blocked_result(
            artifact_contract,
            "Programmer prompt exceeded the context budget.",
        )
        _fill_trace(
            trace,
            raw_output="",
            result=result,
            context_budget=context_budget,
            blocked_before_invoke=True,
            markdown_code_block_found=False,
        )
        return result

    raw_output, attempts = await _invoke_programmer(payload_text)
    content_format = artifact_contract["content_format"]
    code_artifact = _extract_single_output_block(
        raw_output,
        content_format=content_format,
    )
    diagnostics = _artifact_diagnostics(
        code_artifact,
        content_format=content_format,
    )
    if code_artifact and not diagnostics:
        result: WritingProgrammerResult = {
            "artifact_id": artifact_contract["artifact_id"],
            "status": "succeeded",
            "content_format": content_format,
            "code_artifact": code_artifact,
            "diagnostics": [],
        }
    else:
        result = _blocked_result(
            artifact_contract,
            diagnostics[0] if diagnostics
            else "Programmer did not return exactly one valid fenced artifact.",
        )
    _fill_trace(
        trace,
        raw_output=raw_output,
        result=result,
        context_budget=context_budget,
        blocked_before_invoke=False,
        markdown_code_block_found=bool(code_artifact),
        attempts=attempts,
    )
    return result


async def _invoke_programmer(
    payload_text: str,
) -> tuple[str, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    raw_output = ""
    previous_diagnostics: list[str] = []
    for attempt_index in range(3):
        messages = [
            SystemMessage(content=WRITING_PROGRAMMER_PROMPT),
            HumanMessage(content=payload_text),
        ]
        if attempt_index > 0:
            diagnostics_text = "\n".join(
                f"- {diagnostic}" for diagnostic in previous_diagnostics
            )
            retry_prompt = WRITING_PROGRAMMER_RETRY_PROMPT.format(
                diagnostics=diagnostics_text,
            )
            messages.append(HumanMessage(content=retry_prompt))
        timed_out = False
        try:
            response = await asyncio.wait_for(
                _writing_programmer_llm.ainvoke(
                    messages,
                    config=_writing_programmer_llm_config,
                ),
                timeout=PROGRAMMER_LLM_CALL_TIMEOUT_SECONDS,
            )
            raw_output = response.content
        except asyncio.TimeoutError:
            timed_out = True
            raw_output = ""
        content_format = _content_format_from_payload(payload_text)
        code_artifact = _extract_single_output_block(
            raw_output,
            content_format=content_format,
        )
        diagnostics = _artifact_diagnostics(
            code_artifact,
            content_format=content_format,
        )
        attempts.append({
            "attempt": attempt_index + 1,
            "raw_output": raw_output,
            "markdown_code_block_found": bool(code_artifact),
            "diagnostics": diagnostics,
            "timed_out": timed_out,
        })
        if (code_artifact and not diagnostics) or timed_out:
            break
        previous_diagnostics = diagnostics
    return raw_output, attempts


def _extract_single_output_block(
    raw_output: str,
    *,
    content_format: WritingContentFormat,
) -> str:
    if not isinstance(raw_output, str):
        return ""

    match = re.fullmatch(
        r"\s*(?P<fence>`{3,}|~{3,})(?P<language>[A-Za-z0-9_-]*)[ \t]*"
        r"\r?\n(?P<content>.*)\r?\n(?P=fence)[ \t]*\s*",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""

    language = match.group("language")
    content = match.group("content")
    normalized_language = language.strip().casefold()
    if not _language_matches_format(normalized_language, content_format):
        return ""
    code_artifact = content.strip()
    return code_artifact


def _language_matches_format(
    normalized_language: str,
    content_format: WritingContentFormat,
) -> bool:
    allowed_languages = {
        "python": {"python", "py"},
        "markdown": {"markdown", "md"},
        "text": {"text", "txt", "toml", "yaml", "yml", "ini", "cfg", "conf", "env", ""},
        "json": {"json"},
        "csv": {"csv", "text", "txt", ""},
    }
    allowed = allowed_languages[content_format]
    return normalized_language in allowed


def _artifact_diagnostics(
    code_artifact: str,
    *,
    content_format: WritingContentFormat,
) -> list[str]:
    if not code_artifact:
        return []
    lowered = code_artifact.casefold()
    if "notimplementederror" in lowered:
        return ["Programmer output contains NotImplementedError."]
    for line in code_artifact.splitlines():
        stripped = line.strip().casefold()
        if _is_unfinished_comment(stripped):
            return ["Programmer output contains unfinished placeholder text."]
    if content_format == "python":
        if re.search(r"\bexcept\s+(?:Exception|BaseException)\b", code_artifact):
            return ["Programmer output catches a broad exception type."]
        placeholder_diagnostic = _python_placeholder_diagnostic(code_artifact)
        if placeholder_diagnostic:
            return [placeholder_diagnostic]
    if content_format == "markdown" and _markdown_fences_are_unbalanced(
        code_artifact,
    ):
        return ["Programmer output contains unbalanced Markdown code fences."]
    return []


def _is_unfinished_comment(stripped_line: str) -> bool:
    comment_prefixes = ("#", "//", "--", "<!--")
    if not stripped_line.startswith(comment_prefixes):
        return False

    comment_text = stripped_line
    for prefix in comment_prefixes:
        if comment_text.startswith(prefix):
            comment_text = comment_text.removeprefix(prefix).strip()
            break

    unfinished_markers = (
        "todo",
        "fixme",
        "placeholder",
        "implement later",
    )
    has_unfinished_marker = comment_text.startswith(unfinished_markers)
    return has_unfinished_marker


def _python_placeholder_diagnostic(code_artifact: str) -> str:
    """Detect empty Python stubs without rejecting valid control flow."""

    try:
        tree = ast.parse(code_artifact)
    except SyntaxError:
        return ""

    if len(tree.body) == 1:
        only_statement = tree.body[0]
        if isinstance(only_statement, ast.Pass):
            return "Programmer output contains a pass placeholder."
        if _is_ellipsis_statement(only_statement):
            return "Programmer output contains an ellipsis placeholder."

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        if len(node.body) != 1:
            continue
        only_statement = node.body[0]
        if isinstance(only_statement, ast.Pass):
            return "Programmer output contains a pass placeholder."
        if _is_ellipsis_statement(only_statement):
            return "Programmer output contains an ellipsis placeholder."

    return ""


def _is_ellipsis_statement(node: ast.stmt) -> bool:
    """Return whether an AST statement is a standalone ellipsis."""

    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    is_ellipsis = isinstance(value, ast.Constant) and value.value is Ellipsis
    return is_ellipsis


def _markdown_fences_are_unbalanced(text: str) -> bool:
    backtick_count = 0
    tilde_count = 0
    for line in text.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("```"):
            backtick_count += 1
            continue
        if stripped_line.startswith("~~~"):
            tilde_count += 1
    return backtick_count % 2 != 0 or tilde_count % 2 != 0


def _content_format_from_payload(payload_text: str) -> WritingContentFormat:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return "python"
    if not isinstance(payload, dict):
        return "python"
    content_format = payload.get("content_format")
    if content_format in {"python", "markdown", "text", "json", "csv"}:
        return content_format
    return "python"


def _blocked_result(
    artifact_contract: WritingProgrammerContract,
    diagnostic: str,
) -> WritingProgrammerResult:
    result: WritingProgrammerResult = {
        "artifact_id": artifact_contract["artifact_id"],
        "status": "blocked",
        "content_format": artifact_contract["content_format"],
        "code_artifact": "",
        "diagnostics": [diagnostic],
    }
    return result


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    result: WritingProgrammerResult,
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
    markdown_code_block_found: bool,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _writing_programmer_llm_config.route_name
    trace["model"] = _writing_programmer_llm_config.model
    trace["thinking_enabled"] = CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["normalized_output"] = result
    trace["markdown_code_block_found"] = markdown_code_block_found
    if attempts is not None:
        trace["attempts"] = attempts
