"""Bounded proposal and independent review LLM stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import time
import unicodedata

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_identity_proposal_decision,
    validate_identity_review_decision,
)
from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMInvoker,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


IDENTITY_STAGE_ATTEMPT_LIMIT = 3
_IDENTITY_CONTRACT_REGENERATION_PROMPT_TEMPLATES = {
    "proposal": '''\
The prior object failed the closed proposal contract. Return one complete
replacement JSON object. Every proposed_changes item must copy the exact
tagged patch shape: path, value_kind, and exactly one replacement field that
matches value_kind. Re-evaluate the whole object from the original context;
the rejected object is error evidence, not a template to repeat.

Required top-level keys:
{required_top_level_keys}

Exact allowed evidence_ref_ids:
{evidence_ref_ids}

Exact allowed candidate_ids:
{candidate_ids}

Copy any cited identifier exactly from these lists. Use an empty list or null
when the semantic decision does not cite one. Include every required
top-level key and no unknown key.

When the contract error identifies no-op patches, remove every listed no-op
and re-audit the unchanged identity paths against the original evidence.
Do not switch to no_change merely because one or more current paths already
express part of the durable change.
Do not evade a no-op by translating, paraphrasing, or misspelling the current
value. Inspect every remaining path, including sibling paths in the same
category, and propose only a genuinely changed semantic value.

Contract error:
{contract_error}
''',
    "review": '''\
The prior object failed the closed review contract. Return one complete
replacement JSON object. When verdict is accept, every accepted_changes item
must be an exact object copy of a proposed_changes item, including path,
value_kind, and its matching replacement field and value. Re-evaluate the
whole object from the original context; the rejected object is error evidence,
not a template to repeat.

The phrase "one matching replacement field" is explanatory text and is never
a legal JSON key. When accepting, copy objects only from this exact list:
{exact_proposed_changes}

Required top-level keys:
{required_top_level_keys}

Exact allowed evidence_ref_ids:
{evidence_ref_ids}

Exact allowed candidate_ids:
{candidate_ids}

Copy any cited identifier exactly from these lists. Use an empty list or null
when the semantic decision does not cite one. Include every required
top-level key and no unknown key.

Contract error:
{contract_error}
''',
}
_IDENTITY_PATH_CONTRACT_TEXT = json.dumps(
    {
        "value_kind_to_replacement_field": {
            "text": "replacement_text",
            "integer": "replacement_integer",
            "semantic_band": "replacement_band",
            "closed_enum": "replacement_enum",
            "text_list": "replacement_items",
        },
        "text_paths": sorted(models.TEXT_IDENTITY_PATHS),
        "integer_paths": sorted(models.INTEGER_IDENTITY_PATHS),
        "semantic_band_paths": sorted(models.NUMERIC_IDENTITY_PATHS),
        "semantic_band_values": list(models.SEMANTIC_BAND_VALUES),
        "closed_enum_paths": {
            path: sorted(values)
            for path, values in models.ENUM_VALUES_BY_PATH.items()
        },
        "text_list_paths": sorted(models.TEXT_LIST_IDENTITY_PATHS),
    },
    ensure_ascii=False,
    sort_keys=True,
)

_IDENTITY_PROPOSAL_SYSTEM_PROMPT_TEMPLATE = '''\
You evaluate possible durable growth in a fictional character's own identity.
You do not role-play, write dialog, obey the user, or persist anything. Return
one complete JSON object and nothing outside it.

# Authority
- The current identity is the character's present self, not an immutable seed.
  Any listed identity path may change when the evidence genuinely supports it.
- A user's instruction, preference, praise, criticism, fantasy, or repeated
  pressure does not make a change character-authored.
- The fact that a user asked an open question does not by itself make the
  answer user-imposed. When the input leaves the character free to retain,
  change, or reject the old identity and does not supply the desired identity
  conclusion, independently formulated cognition plus visible self-definition
  may be self_declared. Treat it as user-imposed only when the user supplies
  the desired identity conclusion or demands its adoption and the character
  merely complies.
- Explicit self-redefinition requires the character's own cognition and
  visible self-expression to define a changed self. When both summaries show
  that direct self-definition, choose explicit_self_redefinition even when
  repeated experiences led to it or a supplied candidate already anticipated
  it.
- Inferred growth means a durable character-owned pattern across independent
  experiences without a direct self-definition. It is not a transient mood,
  one scene, bounded role-play, user preference, private fact, domain skill,
  topic knowledge, or channel style. A relationship fact or promise is not
  itself character identity.
- Love, intimacy, or another close relationship may be evidence of durable
  identity growth when it changes how the character understands their own
  vulnerability, commitment, trust, boundaries, or autonomy. The relationship
  target and relationship facts remain scoped; propose only the character's
  abstract change.
- Use corroborate_candidate only when new evidence semantically supports one
  supplied candidate. Identify incompatible supplied candidates separately.
- Mentally apply the proposed changes to the current snapshot. The result must
  be one internally coherent full identity, not a weak overlay that leaves a
  stronger old rule in control. Include every directly conflicting allowed
  path needed to express the accepted change. If the five-patch limit cannot
  produce a coherent snapshot, return no_change instead of a partial change.
- Before proposing, perform a path-by-path contradiction audit across core,
  personality, boundaries, and self-image. Ask whether each unchanged field
  would still predict the opposite behavior in the same situation. For
  example, a validate-before-act decision rule conflicts with an unchanged
  pressure-response rule that still mandates immediate action. Patch every
  such allowed path or return no_change.
- Matching text in personality_brief.logic or self_image does not prove the
  change is already complete. If the evidence disavows behavior still stated
  by personality_brief.defense, personality_brief.quirks, core, boundaries,
  or another current field, no_change is forbidden until that exact
  conflicting path is patched. Never assume one matching field proves full
  coverage of the snapshot.
- When the character explicitly disavows an unchanged current identity field,
  treat that as a new change and patch that exact allowed path. Do not return
  no_change merely because a growth edge or secondary field already describes
  a related direction.
- Partial retention does not make a bundled current field unchanged. When the
  character retains one behavior from a field but disavows or replaces another
  behavior in that same field, replace the whole field with one coherent
  description containing the retained and changed parts.

# Global privacy
- Identity is global across private and group contexts, so retain only an
  abstract character-owned change that is safe in every context.
- Never copy participant identity, a quote, intimate detail, channel detail,
  scope identifier, or opaque handle into generated free text.
- Evaluate private_detail_risk by the proposed persisted abstraction and
  generated summaries. Do not mark risk high merely because the source topic
  involves intimacy.
- Cite only supplied evidence_ref_ids and candidate_ids in their dedicated
  identifier fields.
- Mark private_detail_risk=high whenever a safe character-owned abstraction
  cannot preserve the proposed meaning.

# Judgment
- Judge authorship, durability, global applicability, contradiction, and
  privacy semantically. Evidence-card count alone never proves growth.
- Repeated wording or repeated pressure is not independent corroboration.
- proposed_changes are full replacements for supported paths. Use the exact
  tagged value kind required by that path. Numeric values use semantic bands.
- Every proposed patch has exactly three keys. Its only legal shapes are:
  text = path + value_kind="text" + replacement_text;
  integer = path + value_kind="integer" + replacement_integer;
  numeric = path + value_kind="semantic_band" + replacement_band;
  enum = path + value_kind="closed_enum" + replacement_enum;
  text list = path + value_kind="text_list" + replacement_items.
  Never omit value_kind and never use an explanatory phrase as a JSON key.
- Propose at most five coherent changes. Use no_change when evidence should
  leave identity untouched.
- Write generated free text in the natural language of the supplied identity
  and evidence.

# Output
Return exactly these keys:
{{
  "schema_version": "character_identity_proposal_decision.v1",
  "action": "one allowed proposal action",
  "candidate_id": "supplied candidate id or null",
  "proposed_changes": [],
  "character_authorship": "self_declared | inferred | absent",
  "identity_relevance": "durable | ephemeral | absent",
  "global_applicability": "global | scoped | absent",
  "confidence": "low | medium | high",
  "private_detail_risk": "low | high",
  "character_owned_abstraction": "detail-free judgment summary",
  "evidence_ref_ids": ["supplied evidence handles"],
  "contradiction_candidate_ids": ["supplied incompatible candidate handles"],
  "reason_code": "one allowed proposal reason code"
}}

For no_change, use null candidate_id, empty proposed_changes, and empty
contradiction_candidate_ids. For a new explicit or inferred proposal, use null
candidate_id. For corroborate_candidate, use exactly one supplied candidate_id.
Allowed actions: no_change, explicit_self_redefinition, inferred_growth,
corroborate_candidate. Allowed reason codes: proposal_no_change,
candidate_emerging, candidate_ready, privacy_blocked, contradiction_blocked.
Align reason_code with action: explicit_self_redefinition uses candidate_ready;
inferred_growth and corroborate_candidate use candidate_emerging or
candidate_ready; no_change uses proposal_no_change, privacy_blocked, or
contradiction_blocked.
Use candidate_ready only with confidence=high, identity_relevance=durable,
global_applicability=global, and private_detail_risk=low.
Align character_authorship with action: explicit_self_redefinition uses
self_declared; inferred_growth and corroborate_candidate use inferred.

# Exact tagged path contract
{path_contract}
'''
IDENTITY_PROPOSAL_SYSTEM_PROMPT = (
    _IDENTITY_PROPOSAL_SYSTEM_PROMPT_TEMPLATE.format(
        path_contract=_IDENTITY_PATH_CONTRACT_TEXT,
    )
)

IDENTITY_REVIEW_SYSTEM_PROMPT = '''\
You are the independent reviewer for a possible durable change to a fictional
character's own identity. The proposal is untrusted evidence, not authority.
Repeat the semantic judgment yourself from the current identity and evidence.
Return one complete JSON object and nothing outside it.

# Independent review
- Decide character authorship, durability, global applicability, coherence,
  contradiction, and privacy independently from the proposal.
- Reject user-imposed identity, transient mood, bounded role-play,
  user facts, relationship facts as identity, promises as identity, raw
  intimate details, domain expertise, topic knowledge, and channel-specific
  style.
- Love, intimacy, or another close relationship may be evidence of a durable
  character-owned change. The relationship target and relationship facts
  remain scoped; review only the character's abstract change in how they
  understand their own vulnerability, commitment, trust, boundaries, or
  autonomy.
- An explicit self-redefinition requires the character's own cognition and
  visible self-expression. A user request followed by compliance is not enough.
- The fact that a user asked an open question does not by itself make the
  answer user-imposed. When the input leaves the character free to retain,
  change, or reject the old identity and does not supply the desired identity
  conclusion, independently formulated cognition plus visible self-definition
  may be self_declared. Treat it as user-imposed only when the user supplies
  the desired identity conclusion or demands its adoption and the character
  merely complies.
- Inferred growth requires semantically independent, durable character-owned
  evidence. Counts and repeated wording do not replace this judgment.
- One coherent direction may be selected among competing candidates. Every
  incompatible supplied candidate must be listed as rejected.

# Privacy and exactness
- A global identity summary must be safe in both private and group contexts.
- Never include participant identity, quotes, private facts, scope details,
  or opaque handles in generated summary text.
- Evaluate private_detail_risk by the proposed persisted abstraction and
  generated summaries, not merely because the source topic involves intimacy.
- If accepting, copy proposed_changes exactly. Do not improve, rewrite, add,
  remove, or normalize their semantic values.
- Before accepting, mentally apply every proposed patch to the supplied current
  identity. Reject the proposal when any unchanged identity field remains
  directly incompatible with the proposed durable change or would predict the
  opposite behavior. A growth edge or secondary descriptive field alone does
  not override a conflicting core/personality/boundary/self-image rule.
- Perform a path-by-path contradiction audit across every supplied core,
  personality, boundary, and self-image field before accepting. A new
  validate-before-act decision rule remains incoherent when an unchanged
  pressure-response rule still mandates immediate action. Treat that conflict
  as review_rejected instead of assuming one field silently outranks another.
- For a proposed no_change, never assume one matching field proves full
  coverage. In particular, matching personality_brief.logic or self_image
  cannot hide disavowed behavior still stated in personality_brief.defense,
  personality_brief.quirks, core, boundaries, or another current field. If
  such a path remains, the proposal's no_change judgment is semantically
  wrong and the review must reject it.
- Confirm that a directly disavowed unchanged rule receives its own patch.
  Reject no-change reasoning that relies only on the fact that a secondary
  field already points in a similar direction.
- If rejecting or returning no_change, accepted_change_kind is null and
  accepted_changes is empty.
- Write generated free text in the natural language of the supplied identity
  and evidence.

# Output
Return exactly these keys:
{
  "schema_version": "character_identity_review_decision.v1",
  "verdict": "accept | reject | no_change",
  "selected_candidate_id": "supplied selected candidate id or null",
  "rejected_candidate_ids": ["supplied incompatible candidate ids"],
  "accepted_change_kind": "explicit_self_redefinition | inferred_growth | null",
  "accepted_changes": [],
  "character_authorship": "self_declared | inferred | absent",
  "identity_relevance": "durable | ephemeral | absent",
  "coherence": "coherent | conflicting | absent",
  "global_applicability": "global | scoped | absent",
  "review_confidence": "low | medium | high",
  "private_detail_risk": "low | high",
  "character_owned_summary": "detail-free independent judgment",
  "privacy_safe_evidence_summaries": ["detail-free evidence abstractions"],
  "reason_code": "one allowed review reason code"
}

Use no_change only when the proposal also chose no_change. Use reject for a
proposed change that fails review. An acceptance requires at least one
privacy-safe evidence summary.
Allowed reason codes: proposal_no_change, candidate_emerging, candidate_ready,
review_rejected, privacy_blocked, contradiction_blocked.
Align reason_code with verdict: an accepted explicit self-redefinition uses
candidate_ready; other acceptances use candidate_emerging or candidate_ready;
no_change uses proposal_no_change; reject uses review_rejected,
privacy_blocked, or contradiction_blocked.
Use candidate_ready only with review_confidence=high,
identity_relevance=durable, coherence=coherent,
global_applicability=global, and private_detail_risk=low.
For verdict=accept, character_authorship must match accepted_change_kind:
self_declared for explicit_self_redefinition and inferred for inferred_growth.
'''


_PROPOSAL_EXPECTED_FORMAT = json.dumps(
    {
        "schema_version": models.IDENTITY_PROPOSAL_DECISION_SCHEMA_VERSION,
        "action": "no_change",
        "candidate_id": None,
        "proposed_changes": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "global_applicability": "absent",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": "summary",
        "evidence_ref_ids": [],
        "contradiction_candidate_ids": [],
        "reason_code": "proposal_no_change",
    },
    ensure_ascii=False,
)
_REVIEW_EXPECTED_FORMAT = json.dumps(
    {
        "schema_version": models.IDENTITY_REVIEW_DECISION_SCHEMA_VERSION,
        "verdict": "no_change",
        "selected_candidate_id": None,
        "rejected_candidate_ids": [],
        "accepted_change_kind": None,
        "accepted_changes": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "coherence": "absent",
        "global_applicability": "absent",
        "review_confidence": "high",
        "private_detail_risk": "low",
        "character_owned_summary": "summary",
        "privacy_safe_evidence_summaries": [],
        "reason_code": "proposal_no_change",
    },
    ensure_ascii=False,
)


@dataclass(frozen=True)
class IdentityPromptBuildResult:
    """Rendered prompt pair after bounded optional-candidate removal."""

    system_prompt: str
    human_prompt: str
    prompt_chars: int
    candidate_count: int
    evidence_ref_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    evidence_ref_aliases: tuple[tuple[str, str], ...]
    candidate_aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class IdentityStageResult:
    """Sanitized successful semantic-stage result."""

    decision: dict[str, object]
    attempt_count: int
    prompt_chars: int
    output_chars: int
    validation_error_codes: tuple[str, ...]
    trace_id: str


class IdentityPromptBudgetError(ValueError):
    """Raised when required semantic context cannot fit the declared budget."""


class IdentityStageError(RuntimeError):
    """Base typed failure for an identity semantic stage."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        attempt_count: int,
        validation_error_codes: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.attempt_count = attempt_count
        self.validation_error_codes = validation_error_codes


class IdentityStageContractError(IdentityStageError):
    """Raised after bounded full replacements cannot satisfy the schema."""


class IdentityStageProviderError(IdentityStageError):
    """Raised after bounded provider attempts are exhausted."""


_identity_llm = LLInterface()
_identity_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.2,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


def build_identity_proposal_prompt(
    proposal_input: Mapping[str, object],
    *,
    prompt_char_budget: int = models.IDENTITY_PROMPT_CHAR_BUDGET_DEFAULT,
) -> IdentityPromptBuildResult:
    """Render the proposal prompt within the declared character budget."""

    if (
        proposal_input.get("schema_version")
        != models.IDENTITY_PROPOSAL_INPUT_SCHEMA_VERSION
    ):
        raise ValueError("identity proposal input schema_version is invalid")
    return _build_bounded_prompt(
        system_prompt=IDENTITY_PROPOSAL_SYSTEM_PROMPT,
        payload=proposal_input,
        prompt_char_budget=prompt_char_budget,
    )


def build_identity_review_prompt(
    review_input: Mapping[str, object],
    *,
    prompt_char_budget: int = models.IDENTITY_PROMPT_CHAR_BUDGET_DEFAULT,
) -> IdentityPromptBuildResult:
    """Render the independent review prompt within the declared budget."""

    if (
        review_input.get("schema_version")
        != models.IDENTITY_REVIEW_INPUT_SCHEMA_VERSION
    ):
        raise ValueError("identity review input schema_version is invalid")
    return _build_bounded_prompt(
        system_prompt=IDENTITY_REVIEW_SYSTEM_PROMPT,
        payload=review_input,
        prompt_char_budget=prompt_char_budget,
    )


async def propose_identity_growth(
    proposal_input: Mapping[str, object],
    *,
    invoker: LLMInvoker | None = None,
    trace_id: str = "",
    prompt_char_budget: int = models.IDENTITY_PROMPT_CHAR_BUDGET_DEFAULT,
) -> IdentityStageResult:
    """Run the bounded identity proposal semantic stage."""

    prompt = build_identity_proposal_prompt(
        proposal_input,
        prompt_char_budget=prompt_char_budget,
    )
    evidence_ref_ids = set(prompt.evidence_ref_ids)
    candidate_ids = set(prompt.candidate_ids)
    current_identity = proposal_input.get("current_identity")
    if not isinstance(current_identity, Mapping):
        raise ValueError("identity proposal input requires current_identity")

    def validate(
        parsed: Mapping[str, object],
    ) -> models.IdentityProposalDecisionV1:
        decision = validate_identity_proposal_decision(
            parsed,
            evidence_ref_ids=evidence_ref_ids,
            candidate_ids=candidate_ids,
        )
        if decision["action"] != "no_change":
            _require_non_noop_prompt_patches(
                current_identity,
                decision["proposed_changes"],
            )
        return decision

    result = await _run_identity_stage(
        stage="proposal",
        prompt=prompt,
        expected_output_format=_PROPOSAL_EXPECTED_FORMAT,
        validator=validate,
        invoker=invoker or _identity_llm,
        trace_id=trace_id,
    )
    return _restore_identity_stage_result_handles(
        result,
        prompt=prompt,
        stage="proposal",
    )


async def review_identity_growth(
    review_input: Mapping[str, object],
    *,
    invoker: LLMInvoker | None = None,
    trace_id: str = "",
    prompt_char_budget: int = models.IDENTITY_PROMPT_CHAR_BUDGET_DEFAULT,
) -> IdentityStageResult:
    """Run the bounded independent identity review semantic stage."""

    prompt = build_identity_review_prompt(
        review_input,
        prompt_char_budget=prompt_char_budget,
    )
    evidence_ref_ids = set(prompt.evidence_ref_ids)
    candidate_ids = set(prompt.candidate_ids)
    prompt_payload = json.loads(prompt.human_prompt)
    raw_proposal = prompt_payload.get("proposal_decision")
    if not isinstance(raw_proposal, Mapping):
        raise ValueError("identity review input requires proposal_decision")

    def validate(
        parsed: Mapping[str, object],
    ) -> models.IdentityReviewDecisionV1:
        return validate_identity_review_decision(
            parsed,
            proposal=raw_proposal,
            evidence_ref_ids=evidence_ref_ids,
            candidate_ids=candidate_ids,
        )

    result = await _run_identity_stage(
        stage="review",
        prompt=prompt,
        expected_output_format=_REVIEW_EXPECTED_FORMAT,
        validator=validate,
        invoker=invoker or _identity_llm,
        trace_id=trace_id,
    )
    return _restore_identity_stage_result_handles(
        result,
        prompt=prompt,
        stage="review",
    )


def _build_bounded_prompt(
    *,
    system_prompt: str,
    payload: Mapping[str, object],
    prompt_char_budget: int,
) -> IdentityPromptBuildResult:
    """Drop optional older candidates until required context fits."""

    if (
        isinstance(prompt_char_budget, bool)
        or not isinstance(prompt_char_budget, int)
        or prompt_char_budget <= 0
    ):
        raise ValueError("identity prompt_char_budget must be positive")
    (
        bounded_payload,
        evidence_ref_aliases,
        candidate_aliases,
    ) = _alias_prompt_handles(payload)
    raw_candidates = bounded_payload.get("current_candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("identity prompt current_candidates must be a list")
    protected_candidate_ids = _protected_candidate_ids(bounded_payload)
    while True:
        human_prompt = json.dumps(
            bounded_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_chars = len(system_prompt) + len(human_prompt)
        if prompt_chars <= prompt_char_budget:
            evidence_ref_ids, candidate_ids = _input_handles(
                bounded_payload
            )
            return IdentityPromptBuildResult(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                prompt_chars=prompt_chars,
                candidate_count=len(raw_candidates),
                evidence_ref_ids=tuple(sorted(evidence_ref_ids)),
                candidate_ids=tuple(sorted(candidate_ids)),
                evidence_ref_aliases=tuple(
                    sorted(evidence_ref_aliases.items())
                ),
                candidate_aliases=tuple(
                    sorted(candidate_aliases.items())
                ),
            )
        removable_index = _last_removable_candidate_index(
            raw_candidates,
            protected_candidate_ids=protected_candidate_ids,
        )
        if removable_index is not None:
            raw_candidates.pop(removable_index)
            continue
        raise IdentityPromptBudgetError(
            "required identity and evidence exceed the prompt budget"
        )


def _alias_prompt_handles(
    payload: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, str],
    dict[str, str],
]:
    """Replace repository identifiers with short prompt-local handles."""

    aliased = deepcopy(dict(payload))
    raw_cards = aliased.get("evidence_cards")
    raw_candidates = aliased.get("current_candidates")
    if not isinstance(raw_cards, list) or not isinstance(
        raw_candidates,
        list,
    ):
        raise ValueError("identity stage input requires cards and candidates")

    evidence_source_to_alias: dict[str, str] = {}
    evidence_alias_to_source: dict[str, str] = {}
    for index, raw_card in enumerate(raw_cards, start=1):
        source_id = _mapping_text(raw_card, "evidence_ref_id")
        alias = f"evidence-{index}"
        evidence_source_to_alias[source_id] = alias
        evidence_alias_to_source[alias] = source_id
        if not isinstance(raw_card, dict):
            raise ValueError("identity evidence card must be mutable")
        raw_card["evidence_ref_id"] = alias

    candidate_source_to_alias: dict[str, str] = {}
    candidate_alias_to_source: dict[str, str] = {}
    for index, raw_candidate in enumerate(raw_candidates, start=1):
        source_id = _mapping_text(raw_candidate, "candidate_id")
        alias = f"candidate-{index}"
        candidate_source_to_alias[source_id] = alias
        candidate_alias_to_source[alias] = source_id
        if not isinstance(raw_candidate, dict):
            raise ValueError("identity candidate must be mutable")
        raw_candidate["candidate_id"] = alias

    raw_proposal = aliased.get("proposal_decision")
    if isinstance(raw_proposal, dict):
        raw_proposal["evidence_ref_ids"] = _alias_handle_list(
            raw_proposal.get("evidence_ref_ids"),
            source_to_alias=evidence_source_to_alias,
            context="identity proposal evidence refs",
        )
        raw_proposal["candidate_id"] = _alias_optional_handle(
            raw_proposal.get("candidate_id"),
            source_to_alias=candidate_source_to_alias,
            context="identity proposal candidate",
        )
        raw_proposal["contradiction_candidate_ids"] = _alias_handle_list(
            raw_proposal.get("contradiction_candidate_ids"),
            source_to_alias=candidate_source_to_alias,
            context="identity proposal contradiction candidates",
        )

    return (
        aliased,
        evidence_alias_to_source,
        candidate_alias_to_source,
    )


def _alias_handle_list(
    value: object,
    *,
    source_to_alias: Mapping[str, str],
    context: str,
) -> list[str]:
    """Translate a closed list of repository handles into prompt aliases."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    aliases: list[str] = []
    for source_id in value:
        if not isinstance(source_id, str):
            raise ValueError(f"{context} entries must be text")
        alias = source_to_alias.get(source_id)
        if alias is None:
            raise ValueError(f"{context} cites an unknown handle")
        aliases.append(alias)
    return aliases


def _alias_optional_handle(
    value: object,
    *,
    source_to_alias: Mapping[str, str],
    context: str,
) -> str | None:
    """Translate one optional repository handle into a prompt alias."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be text or null")
    alias = source_to_alias.get(value)
    if alias is None:
        raise ValueError(f"{context} cites an unknown handle")
    return alias


def _restore_identity_stage_result_handles(
    result: IdentityStageResult,
    *,
    prompt: IdentityPromptBuildResult,
    stage: str,
) -> IdentityStageResult:
    """Restore validated prompt aliases to repository identifiers."""

    decision = deepcopy(result.decision)
    evidence_aliases = dict(prompt.evidence_ref_aliases)
    candidate_aliases = dict(prompt.candidate_aliases)
    if stage == "proposal":
        decision["evidence_ref_ids"] = _restore_handle_list(
            decision.get("evidence_ref_ids"),
            alias_to_source=evidence_aliases,
            context="identity proposal evidence refs",
        )
        decision["candidate_id"] = _restore_optional_handle(
            decision.get("candidate_id"),
            alias_to_source=candidate_aliases,
            context="identity proposal candidate",
        )
        decision["contradiction_candidate_ids"] = _restore_handle_list(
            decision.get("contradiction_candidate_ids"),
            alias_to_source=candidate_aliases,
            context="identity proposal contradiction candidates",
        )
    elif stage == "review":
        decision["selected_candidate_id"] = _restore_optional_handle(
            decision.get("selected_candidate_id"),
            alias_to_source=candidate_aliases,
            context="identity review selected candidate",
        )
        decision["rejected_candidate_ids"] = _restore_handle_list(
            decision.get("rejected_candidate_ids"),
            alias_to_source=candidate_aliases,
            context="identity review rejected candidates",
        )
    else:
        raise ValueError(f"unknown identity stage: {stage}")
    return IdentityStageResult(
        decision=decision,
        attempt_count=result.attempt_count,
        prompt_chars=result.prompt_chars,
        output_chars=result.output_chars,
        validation_error_codes=result.validation_error_codes,
        trace_id=result.trace_id,
    )


def _restore_handle_list(
    value: object,
    *,
    alias_to_source: Mapping[str, str],
    context: str,
) -> list[str]:
    """Translate validated prompt aliases back to repository handles."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    source_ids: list[str] = []
    for alias in value:
        if not isinstance(alias, str):
            raise ValueError(f"{context} entries must be text")
        source_id = alias_to_source.get(alias)
        if source_id is None:
            raise ValueError(f"{context} cites an unknown prompt handle")
        source_ids.append(source_id)
    return source_ids


def _restore_optional_handle(
    value: object,
    *,
    alias_to_source: Mapping[str, str],
    context: str,
) -> str | None:
    """Translate one optional prompt alias back to a repository handle."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be text or null")
    source_id = alias_to_source.get(value)
    if source_id is None:
        raise ValueError(f"{context} cites an unknown prompt handle")
    return source_id


def _protected_candidate_ids(payload: Mapping[str, object]) -> set[str]:
    """Return proposal-referenced candidates that review must retain."""

    raw_proposal = payload.get("proposal_decision")
    if not isinstance(raw_proposal, Mapping):
        return set()
    protected: set[str] = set()
    candidate_id = raw_proposal.get("candidate_id")
    if isinstance(candidate_id, str) and candidate_id.strip():
        protected.add(candidate_id.strip())
    contradiction_ids = raw_proposal.get("contradiction_candidate_ids")
    if isinstance(contradiction_ids, list):
        protected.update(
            candidate_id.strip()
            for candidate_id in contradiction_ids
            if isinstance(candidate_id, str) and candidate_id.strip()
        )
    return protected


def _last_removable_candidate_index(
    candidates: list[object],
    *,
    protected_candidate_ids: set[str],
) -> int | None:
    """Return the oldest unprotected candidate index, if one exists."""

    for index in range(len(candidates) - 1, -1, -1):
        candidate_id = _mapping_text(candidates[index], "candidate_id")
        if candidate_id not in protected_candidate_ids:
            return index
    return None


def _input_handles(
    stage_input: Mapping[str, object],
) -> tuple[set[str], set[str]]:
    """Return closed evidence and candidate handles from a stage input."""

    raw_cards = stage_input.get("evidence_cards")
    raw_candidates = stage_input.get("current_candidates")
    if not isinstance(raw_cards, list) or not isinstance(
        raw_candidates,
        list,
    ):
        raise ValueError("identity stage input requires cards and candidates")
    evidence_ref_ids = {
        _mapping_text(card, "evidence_ref_id")
        for card in raw_cards
    }
    candidate_ids = {
        _mapping_text(candidate, "candidate_id")
        for candidate in raw_candidates
    }
    return evidence_ref_ids, candidate_ids


def _require_non_noop_prompt_patches(
    current_identity: Mapping[str, object],
    patches: list[models.IdentityPatchV1],
) -> None:
    """Reject replacements equal to the prompt-safe current value."""

    no_op_paths: list[str] = []
    replacement_fields = {
        "text": "replacement_text",
        "integer": "replacement_integer",
        "semantic_band": "replacement_band",
        "closed_enum": "replacement_enum",
        "text_list": "replacement_items",
    }
    for patch in patches:
        path = patch["path"]
        current_value: object = current_identity
        for part in path.split("."):
            if not isinstance(current_value, Mapping):
                raise ValueError(
                    f"identity prompt path is unavailable: {path}"
                )
            current_value = current_value[part]
        replacement_field = replacement_fields[patch["value_kind"]]
        replacement_value = patch.get(replacement_field)
        if _identity_prompt_values_match(
            current_value,
            replacement_value,
            value_kind=patch["value_kind"],
        ):
            no_op_paths.append(path)
    if no_op_paths:
        raise ValueError(
            f"identity patches are no-ops: {sorted(no_op_paths)}"
        )


def _identity_prompt_values_match(
    current_value: object,
    replacement_value: object,
    *,
    value_kind: str,
) -> bool:
    """Compare prompt values while ignoring text-only formatting drift."""

    if value_kind == "text":
        if not isinstance(current_value, str) or not isinstance(
            replacement_value,
            str,
        ):
            return current_value == replacement_value
        return _identity_text_semantics_key(
            current_value
        ) == _identity_text_semantics_key(replacement_value)
    if value_kind == "text_list":
        if not isinstance(current_value, list) or not isinstance(
            replacement_value,
            list,
        ):
            return current_value == replacement_value
        if not all(isinstance(item, str) for item in current_value):
            return current_value == replacement_value
        if not all(isinstance(item, str) for item in replacement_value):
            return current_value == replacement_value
        return [
            _identity_text_semantics_key(item)
            for item in current_value
        ] == [
            _identity_text_semantics_key(item)
            for item in replacement_value
        ]
    return current_value == replacement_value


def _identity_text_semantics_key(value: str) -> str:
    """Remove whitespace and punctuation that cannot constitute growth."""

    return "".join(
        character
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


async def _run_identity_stage(
    *,
    stage: str,
    prompt: IdentityPromptBuildResult,
    expected_output_format: str,
    validator: Callable[
        [Mapping[str, object]],
        Mapping[str, object],
    ],
    invoker: LLMInvoker,
    trace_id: str,
) -> IdentityStageResult:
    """Run one semantic owner with bounded complete replacements."""

    messages = [
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ]
    validation_error_codes: list[str] = []
    effective_trace_id = trace_id or llm_tracing.current_trace_id()
    for attempt_index in range(IDENTITY_STAGE_ATTEMPT_LIMIT):
        started_at = time.perf_counter()
        raw_output = ""
        parsed_output: object = {}
        try:
            response = await invoker.ainvoke(
                messages,
                config=_identity_llm_config,
            )
            raw_output = str(getattr(response, "content", ""))
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            ended_at = time.perf_counter()
            validation_error_codes.append(f"{stage}_provider_error")
            await _record_trace_attempt(
                trace_id=effective_trace_id,
                stage=stage,
                attempt_index=attempt_index,
                messages=messages,
                raw_output="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
            )
            if attempt_index + 1 >= IDENTITY_STAGE_ATTEMPT_LIMIT:
                raise IdentityStageProviderError(
                    f"{stage} provider attempts exhausted",
                    stage=stage,
                    attempt_count=attempt_index + 1,
                    validation_error_codes=tuple(
                        validation_error_codes
                    ),
                ) from exc
            continue

        try:
            parsed_output = parse_llm_json_output(
                raw_output,
                expected_output_format=expected_output_format,
            )
            validated = validator(parsed_output)
        except (KeyError, TypeError, ValueError) as exc:
            ended_at = time.perf_counter()
            validation_error_codes.append(f"{stage}_contract_error")
            await _record_trace_attempt(
                trace_id=effective_trace_id,
                stage=stage,
                attempt_index=attempt_index,
                messages=messages,
                raw_output=raw_output,
                parsed_output=parsed_output,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
            )
            if attempt_index + 1 >= IDENTITY_STAGE_ATTEMPT_LIMIT:
                raise IdentityStageContractError(
                    f"{stage} contract attempts exhausted",
                    stage=stage,
                    attempt_count=attempt_index + 1,
                    validation_error_codes=tuple(
                        validation_error_codes
                    ),
                ) from exc
            regeneration_prompt = (
                _IDENTITY_CONTRACT_REGENERATION_PROMPT_TEMPLATES[
                    stage
                ].format(
                    contract_error=str(exc),
                    required_top_level_keys=json.dumps(
                        sorted(
                            json.loads(expected_output_format).keys()
                        ),
                        ensure_ascii=False,
                    ),
                    evidence_ref_ids=json.dumps(
                        prompt.evidence_ref_ids,
                        ensure_ascii=False,
                    ),
                    candidate_ids=json.dumps(
                        prompt.candidate_ids,
                        ensure_ascii=False,
                    ),
                    exact_proposed_changes=(
                        _review_proposed_changes_json(prompt)
                    ),
                )
            )
            messages = [
                messages[0],
                messages[1],
                AIMessage(content=raw_output),
                HumanMessage(
                    content=regeneration_prompt,
                ),
            ]
            continue

        ended_at = time.perf_counter()
        await _record_trace_attempt(
            trace_id=effective_trace_id,
            stage=stage,
            attempt_index=attempt_index,
            messages=messages,
            raw_output=raw_output,
            parsed_output=validated,
            parse_status="validated",
            status="succeeded",
            started_at=started_at,
            ended_at=ended_at,
        )
        return IdentityStageResult(
            decision=dict(validated),
            attempt_count=attempt_index + 1,
            prompt_chars=prompt.prompt_chars,
            output_chars=len(raw_output),
            validation_error_codes=tuple(validation_error_codes),
            trace_id=effective_trace_id,
        )
    raise AssertionError("identity stage attempt loop did not terminate")


def _review_proposed_changes_json(
    prompt: IdentityPromptBuildResult,
) -> str:
    """Render exact review-copy patch objects from the bounded prompt."""

    payload = json.loads(prompt.human_prompt)
    raw_proposal = payload.get("proposal_decision")
    if not isinstance(raw_proposal, Mapping):
        return "[]"
    raw_changes = raw_proposal.get("proposed_changes")
    if not isinstance(raw_changes, list):
        return "[]"
    return json.dumps(
        raw_changes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


async def _record_trace_attempt(
    *,
    trace_id: str,
    stage: str,
    attempt_index: int,
    messages: list[object],
    raw_output: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    ended_at: float,
) -> None:
    """Record protected prompt/output evidence when a trace is bound."""

    if not trace_id:
        return
    await llm_tracing.record_llm_trace_step(
        trace_id=trace_id,
        stage_name=f"character_identity_growth.{stage}",
        route_name=_identity_llm_config.route_name,
        model_name=_identity_llm_config.model,
        messages=messages,
        response_text=raw_output,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        duration_ms=max(0, int((ended_at - started_at) * 1000)),
        output_state_fields=[],
        sequence=attempt_index,
    )


def _mapping_text(value: object, key: str) -> str:
    """Read one nonempty text field from a stage input object."""

    if not isinstance(value, Mapping):
        raise ValueError("identity stage input row must be an object")
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"identity stage input row requires {key}")
    return text.strip()
