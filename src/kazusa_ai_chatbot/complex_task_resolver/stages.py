"""Production-owned LLM stages for the complex-task resolver."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    WEB_SEARCH_LLM_API_KEY,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MODEL,
    WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    WEB_SEARCH_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

from .constants import STAGE_LLM_TEMPERATURE, STAGE_LLM_TOP_P
from .contracts import ComplexTaskValidationError
from .subagent import _SUBAGENT_TOOLS_TEXT

_PLANNER_PROMPT = '''\
You split a complicated user question into a short semantic task list.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Task
Read the user question and compact context. Return the smallest task list that
lets later stages answer bottom-up. Planning only decomposes. Do not answer
the question and do not cite sources.

# Output Format
{
  "tasks": [
    {
      "objective": "one concrete subtask objective",
      "kind": "subtask|evidence_need|algorithmic_task|synthesis"
    }
  ]
}

# Rules
- Return 2 to 6 tasks.
- Use "evidence_need" only for public external facts, public source comparison,
  current public product/version/support facts, or source-bound public evidence.
- When a question depends on a diff, repository-local code, private plan, prior
  conversation, local artifact, or project-internal term that is not supplied in
  context, create a subtask that records the missing supplied artifact and a
  synthesis task for bounded response; keep that work outside evidence_need.
- Use "algorithmic_task" only for deterministic arithmetic or numeric scoring
  that can be expressed as a numeric expression with visible operands.
- For an "algorithmic_task", include the numeric inputs and already-stated
  units needed for the calculation in the objective. Do not make later stages
  recover hidden operands from a vague label.
- Semantic comparison, feasibility review, collapse review, recommendation,
  route selection, and source-quality judgment are not calculation work unless
  a concrete numeric formula with visible operands is supplied.
- Counting qualitative differences between requirements is semantic comparison,
  not calculation work, unless the caller supplies an explicit numeric scoring
  formula.
- Use "synthesis" only for final bottom-up consolidation.
- When a question needs the same public fact, version, support detail, or source
  confirmation for several independent named targets, create one evidence_need
  task per target before the synthesis or recommendation task.
- Output only the requested semantic task list. Leave graph bookkeeping,
  evidence references, dependencies, inputs, outputs, and operational metadata
  to deterministic code.
'''

_planner_llm = LLInterface()
_planner_llm_config = LLMCallConfig(
    stage_name="complex_task_resolver.graph_planner",
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=WEB_SEARCH_LLM_THINKING_ENABLED),
)


async def plan_complex_task_graph(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return a semantic task decomposition for the current resolver run."""

    response = await _planner_llm.ainvoke(
        [
            SystemMessage(content=_PLANNER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_planner_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict) or not parsed:
        raise ComplexTaskValidationError("planner stage: empty parsed output")
    result = parsed
    return result


_NODE_RESOLVER_PROMPT_TEMPLATE = '''\
You are resolving exactly one active complex-task graph node.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Available Resolver-Local Subagents
{subagent_tools}

# Output Format
Return one of these semantic decision shapes.

To split a mixed node into children:
{{"decision": "expand", "children": [{{"objective": "one child task", "work_type": "subtask|public_evidence|calculation|synthesis", "reason": "why this child is needed"}}]}}

To record semantic knowledge that can be consolidated:
{{"decision": "record_knowledge", "investigation_summary": "...", "knowledge_we_know_so_far": [], "knowledge_still_lacking": [], "recommended_next_iteration": [], "evidence_boundary_notes": [], "completion": "completed|blocked|not_answerable", "continuation_tasks": [{{"objective": "bounded executable child task", "work_type": "subtask|public_evidence|calculation|synthesis", "reason": "why this task is needed"}}]}}

To ask one resolver-local subagent for help:
{{"decision": "use_subagent", "capability": "subagent name from the list above", "action": "supported action name", "objective": "bounded objective for this node", "request": {{}}, "requirements": {{}}}}

When a specific local next pass would improve the node but no child task or
subagent request is ready:
{{"decision": "continue_locally", "action": "short semantic action", "result_summary": "what this pass learned", "blockers": ["specific blocker"], "next_action": "what the next pass should try", "continuation_tasks": [{{"objective": "bounded executable child task", "work_type": "subtask|public_evidence|calculation|synthesis", "reason": "why this task is needed"}}]}}

# Algorithmic Payload Shape
For arithmetic, use capability "algorithmic" and action "evaluate_expression".
The request must be:
{{"expression": "caller-prepared numeric expression", "label": "short_result_label", "input_values": [{{"label": "short_operand_name", "value": "numeric text exactly used in expression", "source_text": "exact text from the provided node projection containing the value"}}], "formula_constants": [{{"value": "numeric text exactly used in expression", "purpose": "why this constant belongs to the formula"}}]}}

The expression may use numbers, arithmetic operators, safe math functions such
as sqrt/round/min/max/sum, Decimal("1.2"), Fraction(1, 3), math.<public_name>,
and statistics.<public_name>. Do not put units in the expression.
Every numeric literal in the expression must be declared either in
input_values or formula_constants. input_values must quote source_text already
present in the supplied active node, parent chain, or sibling summaries. Use
formula_constants only for mathematical constants or formula constants such as
100 in a percentage calculation. Do not invent missing operands. If a needed
operand is missing from the supplied material, record it in
knowledge_still_lacking; record missing operands instead of asking the
calculator to infer them.

# Rules
- Populate semantic projection fields the same way the final packet does: what
  this node investigated, what is known so far, what is still lacking, the next
  semantic direction if one exists, and evidence/tool/source boundaries.
- Use expand only when the active node needs two or more narrower executable
  child tasks. Each child objective should be more specific than the active
  node and ready for evidence, calculation, direct semantic recording, or
  synthesis work.
- For a bounded public_evidence leaf, use the evidence capability. When a
  public evidence node still bundles independent targets or fact dimensions,
  create one narrower public_evidence child for each source-oriented request
  that can be investigated independently.
- Use the evidence capability only for public external source questions. For a
  private or caller-supplied artifact such as a diff, local code, internal plan,
  prior conversation, or project-specific interface that is absent from the
  active node, root question, or context, record the missing artifact and the
  specific supplied context needed for a bounded answer.
- For design, planning, documentation, or architecture leaves answerable from
  the supplied task and context, record semantic knowledge directly.
  Missing implementation preferences should become reviewable assumptions,
  knowledge_still_lacking rows, or recommended_next_iteration rows rather than
  marking the node not answerable. Use completion "not_answerable" only when
  the objective itself is outside resolver scope or cannot be handled safely.
- If active_node.attempts is non-empty, use those prior attempts to avoid
  repeating the same blocked action. Choose a different valid output shape or
  block explicitly if the blocker cannot be solved inside this module.
- Semantic comparison, feasibility review, collapse review, recommendation,
  route selection, and source-quality judgment are not calculation work unless
  a concrete numeric formula with visible operands is supplied.
- Counting qualitative differences between requirements is semantic comparison,
  not calculation work; record the differing dimensions as semantic knowledge
  unless an explicit numeric scoring formula is supplied.
- If the active node requires multiple independent numeric outputs, return
  an expand decision into smaller calculation children instead of forcing one
  expression to stand for every output.
- If the input stage is "subagent_request_repair", return exactly one
  use_subagent decision for required_subagent when the needed inputs are present.
  If required inputs are missing, record knowledge explaining the
  missing input.
- If the active node asks for current product facts, use the evidence
  capability.
- For current public version, release, download, documentation, support, or
  availability facts, make the evidence request source-oriented: put the as-of
  date in request or requirements, and make request.query an official source page,
  canonical URL, or short official-page query for the named target.
- For exact latest downloadable version numbers, prefer the target's official
  download or current-release page as the source. Use lifecycle, roadmap, or
  release-policy pages as support evidence.
- For an algorithmic_task node with the required numbers available in the
  active node, root question, parent chain, or sibling summaries, do not return
  record_knowledge with direct arithmetic. Use the algorithmic capability with
  action "evaluate_expression".
- Prefer a subagent request over direct arithmetic or current-source claims.
- For energy arithmetic, first interpret and normalize units in the active
  node/root question, then pass only the numeric expression. If unit meaning is
  ambiguous, record the ambiguity instead of guessing.
- For shorthand numeric units such as "6k tokens", keep the visible operand
  value from the source text as "6" and put the multiplier "1000" in
  formula_constants. Example: "6k input" becomes expression "6 * 1000",
  input_values value "6" with source_text containing "6k input", and
  formula_constants value "1000" with purpose "k means thousand tokens".
- For clock-time arithmetic, first normalize times into numeric minutes from
  midnight before calling the algorithmic subagent. Example: 7:00 PM becomes
  1140, 9:30 PM becomes 1290, and a schedule finishing 140 minutes after
  7:00 PM uses expression "1140 + 140". Put "1140", "1290", and "140" in
  input_values or formula_constants as plain numeric text; never put
  "7 * 60" or "7:00 PM" in input_values.value.
- Do not call a subagent for pure synthesis or safety wording.
- Use continuation_tasks only for resolver-owned executable child tasks. Do
  not put final dialog decisions, persona judgment, user clarification,
  approval, scheduler work, adapter work, shell work, filesystem work, or
  generic tool calls in continuation_tasks.
- recommended_next_iteration is semantic guidance for cognition and review
  only. It is never an executable instruction and must not duplicate executable
  child work as prose commands.
- Output only semantic decision content. Leave graph bookkeeping, operational
  metadata, execution logs, lower-layer storage details, and internal
  contract envelopes to deterministic code.
- Do not add unsupported fields.
'''

_NODE_RESOLVER_PROMPT = _NODE_RESOLVER_PROMPT_TEMPLATE.format(
    subagent_tools=_SUBAGENT_TOOLS_TEXT
)

_node_resolver_llm = LLInterface()
_node_resolver_llm_config = LLMCallConfig(
    stage_name="complex_task_resolver.active_node_resolver",
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=WEB_SEARCH_LLM_THINKING_ENABLED),
)


async def resolve_complex_task_node(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return one active-node resolution, expansion, or subagent request."""

    response = await _node_resolver_llm.ainvoke(
        [
            SystemMessage(content=_NODE_RESOLVER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_node_resolver_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict) or not parsed:
        raise ComplexTaskValidationError(
            "node resolver stage: empty parsed output"
        )
    result = parsed
    return result


_COLLAPSE_PROMPT = '''\
You review whether one active task duplicates one existing resolved candidate.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Task
Use only the active node and candidate list. If one candidate already answers
the active node, identify it by semantic text from the candidate. Otherwise do
not collapse.

# Output Format
{
  "collapse_decision": {
    "should_collapse": true,
    "matching_candidate": "copy a distinctive objective or summary phrase from the candidate",
    "reason": "short observable reason"
  }
}

When no collapse is justified, return:
{
  "collapse_decision": {
    "should_collapse": false,
    "matching_candidate": "",
    "reason": "not semantically duplicate"
  }
}

# Rules
- matching_candidate must be copied from candidate semantic text when
  should_collapse is true.
- Output only semantic duplicate-review content. Leave graph bookkeeping,
  graph events, operational metadata, execution logs, lower-layer storage
  details, and internal contract envelopes to deterministic code.
'''

_collapse_llm = LLInterface()
_collapse_llm_config = LLMCallConfig(
    stage_name="complex_task_resolver.collapse_review",
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=WEB_SEARCH_LLM_THINKING_ENABLED),
)


async def review_complex_task_collapse(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return one bounded collapse decision for the active node."""

    response = await _collapse_llm.ainvoke(
        [
            SystemMessage(content=_COLLAPSE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_collapse_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict) or not parsed:
        raise ComplexTaskValidationError("collapse stage: empty parsed output")
    result = parsed
    return result


_SYNTHESIZER_PROMPT = '''\
You are producing a structural investigation packet, not character dialog.
Return one JSON object only.
Do not include hidden chain-of-thought.
Synthesize bottom-up from resolved and unresolved node summaries.
Each node summary contains the same semantic projection fields as the final
packet. Use those fields instead of inventing an answer from operational
metadata.
For evidence nodes, investigation_summary may contain mixed observations,
proxy matches, and missing coverage. Do not copy an entire prose summary into
knowledge_we_know_so_far; preserve caveats and put missing or weak coverage in
knowledge_still_lacking.
Calculation facts are known only when they appear in resolved calculation
summaries supplied to you. If a blocked calculation branch names a desired
number or formula, do not calculate it during synthesis; preserve the missing
numeric result in knowledge_still_lacking.
Do not judge whether the user's full question is answered.
Do not decide whether the character should speak, ask the user, retry, or stop.
Only consolidate semantic knowledge for the later cognition stage to judge.
Use recommended_next_iteration for useful evidence directions only. Do not
turn every limitation into a research instruction.

# Output Format
{
  "investigation_summary": "concise summary of the knowledge collected",
  "knowledge_we_know_so_far": ["semantic knowledge found so far"],
  "knowledge_still_lacking": ["specific missing or weak knowledge"],
  "recommended_next_iteration": [
    "semantic next evidence target, only when it would be meaningfully narrower"
  ],
  "evidence_boundary_notes": [
    "source, confidence, tool, or execution-boundary notes for cognition to judge"
  ],
  "continuation_tasks": [
    {
      "objective": "bounded executable root-level follow-up task",
      "work_type": "subtask|public_evidence|calculation|synthesis",
      "reason": "why this task should run before final packet return"
    }
  ]
}

# Follow-Up Rules
- Omit continuation_tasks or return an empty list when no resolver-owned
  continuation is needed.
- Use continuation_tasks only for public evidence, algorithmic, decomposition,
  or synthesis work this resolver can execute inside its graph limits.
- Do not put user clarification, unavailable external access, persona
  judgment, final dialog decisions, scheduler work, adapter work, shell work,
  filesystem work, or generic tool calls in continuation_tasks.
- recommended_next_iteration remains semantic guidance for cognition and
  review only. It is never an executable instruction.
- Output only semantic synthesis content. Leave graph bookkeeping,
  operational metadata, execution logs, lower-layer storage details, and
  internal contract envelopes to deterministic code.
'''

_synthesizer_llm = LLInterface()
_synthesizer_llm_config = LLMCallConfig(
    stage_name="complex_task_resolver.bottom_up_synthesis",
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=WEB_SEARCH_LLM_THINKING_ENABLED),
)


async def synthesize_complex_task_packet(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return the bottom-up packet fields for the current resolver run."""

    response = await _synthesizer_llm.ainvoke(
        [
            SystemMessage(content=_SYNTHESIZER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_synthesizer_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict) or not parsed:
        raise ComplexTaskValidationError("synthesizer stage: empty parsed output")
    result = parsed
    return result
