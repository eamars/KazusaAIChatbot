"""Bounded proposal and independent review LLM stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import time
import unicodedata

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.validation import (
    IdentityContractViolation,
    validate_identity_proposal_wire,
    validate_identity_review_wire,
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
IDENTITY_REPAIR_DESCRIPTOR_CHAR_LIMIT = 800

_IDENTITY_PATH_CONTRACT_TEXT = "\n".join((
    "Text paths (replacement is a string): "
    + ", ".join(sorted(models.TEXT_IDENTITY_PATHS)),
    "Integer paths (replacement is an integer): "
    + ", ".join(sorted(models.INTEGER_IDENTITY_PATHS)),
    "Semantic-band paths (replacement is one band): "
    + ", ".join(sorted(models.NUMERIC_IDENTITY_PATHS)),
    "Allowed semantic bands: "
    + ", ".join(sorted(models.SEMANTIC_BAND_VALUES)),
    "Closed-enum paths and their allowed values:",
    "; ".join(
        f"{path}={sorted(values)}"
        for path, values in sorted(models.ENUM_VALUES_BY_PATH.items())
    ),
    "Text-list paths (replacement is a list of strings): "
    + ", ".join(sorted(models.TEXT_LIST_IDENTITY_PATHS)),
))

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
- When evidence explicitly says that chosen vulnerability, reciprocal care,
  or love changed who the character understands themselves to be, treat that
  as direct self-redefinition when the current identity still separates
  intimacy from self or still predicts retreat. Patch the contradictory
  identity path; do not return no_change merely because a growth edge mentions
  openness in general.
- Use corroborate_candidate only when new evidence semantically supports one
  supplied candidate. Identify incompatible supplied candidates separately.
- A supplied candidate is a semantic hypothesis, not a promotion decision. One
  fresh independent root can justify corroborate_candidate when it supports
  the candidate's direction; do not apply deterministic cadence or promotion
  thresholds in this stage. Policy will hold an accepted candidate until its
  roots and dates mature.
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
- No_change is valid only when the evidence leaves every directly relevant
  current identity path coherent or adds no durable identity meaning. A direct
  disavowal of a current path is a change even when another field already
  mentions a related capacity.
- Partial retention does not make a bundled current field unchanged. When the
  character retains one behavior from a field but disavows or replaces another
  behavior in that same field, replace the whole field with one coherent
  description containing the retained and changed parts.
{corroborate_guidance}

# Global privacy
- Identity is global across private and group contexts, so retain only an
  abstract character-owned change that is safe in every context.
- Never copy participant identity, a quote, intimate detail, channel detail,
  scope identifier, or opaque handle into generated free text.
- Evaluate private_detail_risk by the proposed persisted abstraction and
  generated summaries. Do not mark risk high merely because the source topic
  involves intimacy.
- Use only one-based evidence and candidate row indices from this prompt;
  never emit or copy repository identifiers.
- Mark private_detail_risk=high whenever a safe character-owned abstraction
  cannot preserve the proposed meaning.

# Judgment
- Judge authorship, durability, global applicability, contradiction, and
  privacy semantically. Evidence-card count alone never proves growth.
- Repeated wording or repeated pressure is not independent corroboration.
- proposed_changes are full replacements for supported paths. Each patch has
  exactly two keys: path plus replacement. The replacement type is fixed by
  the path in the exact path/type registry below.
- Do not emit value_kind or any replacement-key variant; the path determines
  the expected type.
- Propose at most five coherent changes. Use no_change when evidence should
  leave identity untouched.
- Write generated free text in the natural language of the supplied identity
  and evidence.

# Output
Return exactly these keys:
{{
  "action": "one allowed proposal action",
  "candidate_index": null,
  "proposed_changes": [{{"path": "one allowed identity path", "replacement": "typed replacement"}}],
  "character_authorship": "self_declared | inferred | absent",
  "identity_relevance": "durable | ephemeral | absent",
  "global_applicability": "global | scoped | absent",
  "confidence": "low | medium | high",
  "private_detail_risk": "low | high",
  "character_owned_abstraction": "detail-free judgment summary",
  "evidence_indices": [1],
  "contradiction_candidate_indices": []
}}

For no_change, use null candidate_index, empty proposed_changes, and empty
contradiction_candidate_indices. For a new explicit or inferred proposal, use
null candidate_index. For corroborate_candidate, use exactly one valid
candidate_index. Evidence and contradiction lists contain only one-based
indices of rows shown in this prompt. A no-op replacement that matches the
current identity value is invalid; return no_change instead when evidence
adds no materially new durable identity meaning.
For no_change, character_owned_abstraction is still required: provide a short
nonempty detail-free reason for the judgment and never use null.
For no_change, the semantic dimensions may describe a character-owned change
that is already represented in the current identity; no candidate or revision
is created. Keep the action, candidate index, and patch list consistent with
no_change.

# Exact path/type contract
{path_contract}
'''

_IDENTITY_PROPOSAL_SYSTEM_PROMPT_NO_CORROBORATE = (
    _IDENTITY_PROPOSAL_SYSTEM_PROMPT_TEMPLATE.format(
        corroborate_guidance=(
            "- No current candidate rows are shown. "
            "corroborate_candidate is unavailable; choose no_change, "
            "explicit_self_redefinition, or inferred_growth."
        ),
        path_contract=_IDENTITY_PATH_CONTRACT_TEXT,
    )
)
_IDENTITY_PROPOSAL_SYSTEM_PROMPT_WITH_CORROBORATE = (
    _IDENTITY_PROPOSAL_SYSTEM_PROMPT_TEMPLATE.format(
        corroborate_guidance=(
            "- corroborate_candidate is allowed only when the prompt lists "
            "current candidates; it requires exactly one valid candidate_index "
            "from the visible candidate rows."
        ),
        path_contract=_IDENTITY_PATH_CONTRACT_TEXT,
    )
)
IDENTITY_PROPOSAL_SYSTEM_PROMPT = (
    _IDENTITY_PROPOSAL_SYSTEM_PROMPT_WITH_CORROBORATE
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
- A review may accept corroboration before promotion readiness. When a supplied
  candidate and one fresh root express the same character-owned direction,
  judge the semantic support and leave cadence to deterministic policy.
- One coherent direction may be selected among competing candidates. Every
  incompatible supplied candidate must be listed as rejected.

# Privacy and exactness
- A global identity summary must be safe in both private and group contexts.
- Never include participant identity, quotes, private facts, scope details,
  or opaque handles in generated summary text.
- Evaluate private_detail_risk by the proposed persisted abstraction and
  generated summaries, not merely because the source topic involves intimacy.
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
- Write generated free text in the natural language of the supplied identity
  and evidence.

# Output
Return exactly these keys:
{
  "verdict": "accept | reject | no_change",
  "selected_candidate_index": null,
  "rejected_candidate_indices": [],
  "character_authorship": "self_declared | inferred | absent",
  "identity_relevance": "durable | ephemeral | absent",
  "coherence": "coherent | conflicting | absent",
  "global_applicability": "global | scoped | absent",
  "review_confidence": "low | medium | high",
  "private_detail_risk": "low | high",
  "character_owned_summary": "detail-free independent judgment",
  "privacy_safe_evidence_summaries": ["detail-free evidence abstractions"],
}

Use no_change only when the proposal also chose no_change. Use reject for a
proposed change that fails review. An acceptance requires at least one
privacy-safe evidence summary.
For no_change or reject, character_owned_summary is still required: provide a
short nonempty detail-free judgment and never use null. Use an empty
privacy_safe_evidence_summaries list when no accepted evidence summary is
warranted.
Selected and rejected candidate lists contain only one-based indices of the
visible candidate rows. The accepted patches are the proposal patches already
listed in this prompt; do not serialize accepted changes again.
When no candidate rows are shown, selected_candidate_index must be null and
rejected_candidate_indices must be an empty list; never use 1 as a placeholder.
'''


_PROPOSAL_EXPECTED_FORMAT = json.dumps(
    {
        "action": "no_change",
        "candidate_index": None,
        "proposed_changes": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "global_applicability": "absent",
        "confidence": "high",
        "private_detail_risk": "low",
        "character_owned_abstraction": (
            "No durable identity change is established."
        ),
        "evidence_indices": [],
        "contradiction_candidate_indices": [],
    },
    ensure_ascii=False,
)
_REVIEW_EXPECTED_FORMAT = json.dumps(
    {
        "verdict": "no_change",
        "selected_candidate_index": None,
        "rejected_candidate_indices": [],
        "character_authorship": "absent",
        "identity_relevance": "absent",
        "coherence": "absent",
        "global_applicability": "absent",
        "review_confidence": "high",
        "private_detail_risk": "low",
        "character_owned_summary": "No durable identity change is established.",
        "privacy_safe_evidence_summaries": [],
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
        system_prompt_builder=_proposal_system_prompt,
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
    evidence_ref_ids = prompt.evidence_ref_ids
    candidate_ids = prompt.candidate_ids
    current_identity = proposal_input.get("current_identity")
    if not isinstance(current_identity, Mapping):
        raise ValueError("identity proposal input requires current_identity")

    def validate(
        parsed: Mapping[str, object],
    ) -> models.IdentityProposalDecisionV1:
        decision = validate_identity_proposal_wire(
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
    evidence_ref_ids = prompt.evidence_ref_ids
    candidate_ids = prompt.candidate_ids
    prompt_payload = json.loads(prompt.human_prompt)
    raw_proposal = prompt_payload.get("proposal_decision")
    if not isinstance(raw_proposal, Mapping):
        raise ValueError("identity review input requires proposal_decision")
    raw_candidates = prompt_payload.get("current_candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("identity review input requires current_candidates")
    candidate_source_by_index = {
        index: source_id
        for index, source_id in prompt.candidate_aliases
    }
    candidate_change_kinds = {
        candidate_source_by_index[candidate["candidate_index"]]: (
            candidate["change_kind"]
        )
        for candidate in raw_candidates
    }

    def validate(
        parsed: Mapping[str, object],
    ) -> models.IdentityReviewDecisionV1:
        return validate_identity_review_wire(
            parsed,
            proposal=raw_proposal,
            evidence_ref_ids=evidence_ref_ids,
            candidate_ids=candidate_ids,
            candidate_change_kinds=candidate_change_kinds,
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
    system_prompt_builder: Callable[[int], str] | None = None,
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
    bounded_payload = deepcopy(dict(payload))
    raw_candidates = bounded_payload.get("current_candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("identity prompt current_candidates must be a list")
    protected_candidate_ids = _protected_candidate_ids(bounded_payload)
    while True:
        effective_system_prompt = (
            system_prompt_builder(len(raw_candidates))
            if system_prompt_builder is not None
            else system_prompt
        )
        (
            prompt_payload,
            evidence_ref_aliases,
            candidate_aliases,
        ) = _normalize_prompt_rows(bounded_payload)
        human_prompt = json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prompt_chars = len(effective_system_prompt) + len(human_prompt)
        if prompt_chars <= prompt_char_budget:
            return IdentityPromptBuildResult(
                system_prompt=effective_system_prompt,
                human_prompt=human_prompt,
                prompt_chars=prompt_chars,
                candidate_count=len(raw_candidates),
                evidence_ref_ids=tuple(
                    source_id
                    for _, source_id in evidence_ref_aliases
                ),
                candidate_ids=tuple(
                    source_id
                    for _, source_id in candidate_aliases
                ),
                evidence_ref_aliases=tuple(
                    sorted(evidence_ref_aliases)
                ),
                candidate_aliases=tuple(
                    sorted(candidate_aliases)
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


def _proposal_system_prompt(candidate_count: int) -> str:
    """Render the proposal system prompt for the visible candidate set."""

    if candidate_count > 0:
        return _IDENTITY_PROPOSAL_SYSTEM_PROMPT_WITH_CORROBORATE
    return _IDENTITY_PROPOSAL_SYSTEM_PROMPT_NO_CORROBORATE


def _normalize_prompt_rows(
    payload: Mapping[str, object],
) -> tuple[
    dict[str, object],
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    """Build prompt-visible rows with one-based indices and uniform patches."""

    aliased = deepcopy(dict(payload))
    raw_cards = aliased.get("evidence_cards")
    raw_candidates = aliased.get("current_candidates")
    if not isinstance(raw_cards, list) or not isinstance(
        raw_candidates,
        list,
    ):
        raise ValueError("identity stage input requires cards and candidates")

    evidence_ref_aliases: list[tuple[int, str]] = []
    candidate_aliases: list[tuple[int, str]] = []
    prompt_cards: list[dict[str, object]] = []
    prompt_candidates: list[dict[str, object]] = []
    for index, raw_card in enumerate(raw_cards, start=1):
        source_id = _mapping_text(raw_card, "evidence_ref_id")
        if not isinstance(raw_card, dict):
            raise ValueError("identity evidence card must be mutable")
        prompt_card = deepcopy(dict(raw_card))
        prompt_card.pop("evidence_ref_id", None)
        prompt_card.pop("schema_version", None)
        prompt_card["evidence_index"] = index
        evidence_ref_aliases.append((index, source_id))
        prompt_cards.append(prompt_card)

    for index, raw_candidate in enumerate(raw_candidates, start=1):
        source_id = _mapping_text(raw_candidate, "candidate_id")
        if not isinstance(raw_candidate, dict):
            raise ValueError("identity candidate must be mutable")
        prompt_candidate = deepcopy(dict(raw_candidate))
        prompt_candidate.pop("candidate_id", None)
        prompt_candidate.pop("schema_version", None)
        prompt_candidate["candidate_index"] = index
        prompt_candidate["proposed_changes"] = _normalize_prompt_patches(
            raw_candidate.get("proposed_changes"),
            context=f"current_candidates[{index - 1}]",
        )
        candidate_aliases.append((index, source_id))
        prompt_candidates.append(prompt_candidate)

    raw_proposal = aliased.get("proposal_decision")
    prompt_proposal = None
    if isinstance(raw_proposal, Mapping):
        prompt_proposal = _normalize_prompt_proposal(
            raw_proposal,
            evidence_ref_aliases=evidence_ref_aliases,
            candidate_aliases=candidate_aliases,
        )
    prompt_payload: dict[str, object] = {
        "current_identity": deepcopy(aliased.get("current_identity")),
        "evidence_cards": prompt_cards,
        "current_candidates": prompt_candidates,
    }
    if prompt_proposal is not None:
        prompt_payload["proposal_decision"] = prompt_proposal
    return (
        prompt_payload,
        evidence_ref_aliases,
        candidate_aliases,
    )


def _normalize_prompt_patches(
    value: object,
    *,
    context: str,
) -> list[dict[str, object]]:
    """Normalize tagged internal patches to uniform path/replacement rows."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{context} proposed_changes must be a list")
    normalized: list[dict[str, object]] = []
    for index, raw_patch in enumerate(value):
        if not isinstance(raw_patch, Mapping):
            raise ValueError(f"{context} proposed_changes[{index}] must be an object")
        path = _mapping_text(raw_patch, "path")
        replacement_key = {
            "text": "replacement_text",
            "integer": "replacement_integer",
            "semantic_band": "replacement_band",
            "closed_enum": "replacement_enum",
            "text_list": "replacement_items",
        }[raw_patch["value_kind"]]
        normalized.append({
            "path": path,
            "replacement": deepcopy(raw_patch[replacement_key]),
        })
    return normalized


def _normalize_prompt_proposal(
    proposal: Mapping[str, object],
    *,
    evidence_ref_aliases: list[tuple[int, str]],
    candidate_aliases: list[tuple[int, str]],
) -> dict[str, object]:
    """Normalize the prompt-visible proposal to wire indices and patches."""

    evidence_index_by_source = {
        source_id: index
        for index, source_id in evidence_ref_aliases
    }
    candidate_index_by_source = {
        source_id: index
        for index, source_id in candidate_aliases
    }
    raw_evidence_ids = proposal.get("evidence_ref_ids")
    if not isinstance(raw_evidence_ids, list):
        raise ValueError("identity proposal evidence_ref_ids must be a list")
    evidence_indices = [
        evidence_index_by_source[source_id]
        for source_id in raw_evidence_ids
    ]
    raw_candidate_id = proposal.get("candidate_id")
    candidate_index = None
    if isinstance(raw_candidate_id, str) and raw_candidate_id.strip():
        candidate_index = candidate_index_by_source[raw_candidate_id.strip()]
    raw_contradictions = proposal.get("contradiction_candidate_ids")
    if not isinstance(raw_contradictions, list):
        raise ValueError(
            "identity proposal contradiction_candidate_ids must be a list"
        )
    contradiction_indices = [
        candidate_index_by_source[source_id]
        for source_id in raw_contradictions
    ]
    normalized: dict[str, object] = {
        "action": proposal["action"],
        "candidate_index": candidate_index,
        "proposed_changes": _normalize_prompt_patches(
            proposal.get("proposed_changes"),
            context="proposal_decision",
        ),
        "character_authorship": proposal["character_authorship"],
        "identity_relevance": proposal["identity_relevance"],
        "global_applicability": proposal["global_applicability"],
        "confidence": proposal["confidence"],
        "private_detail_risk": proposal["private_detail_risk"],
        "character_owned_abstraction": proposal[
            "character_owned_abstraction"
        ],
        "evidence_indices": evidence_indices,
        "contradiction_candidate_indices": contradiction_indices,
    }
    return normalized


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
    evidence_source_to_index = {
        source_id: index
        for index, source_id in evidence_aliases.items()
    }
    candidate_source_to_index = {
        source_id: index
        for index, source_id in candidate_aliases.items()
    }
    if stage == "proposal":
        decision["evidence_ref_ids"] = _restore_handle_list(
            _source_ids_to_indices(
                decision.get("evidence_ref_ids"),
                source_to_index=evidence_source_to_index,
                context="identity proposal evidence refs",
            ),
            alias_to_source=evidence_aliases,
            context="identity proposal evidence refs",
        )
        decision["candidate_id"] = _restore_optional_handle(
            _optional_source_id_to_index(
                decision.get("candidate_id"),
                source_to_index=candidate_source_to_index,
                context="identity proposal candidate",
            ),
            alias_to_source=candidate_aliases,
            context="identity proposal candidate",
        )
        decision["contradiction_candidate_ids"] = _restore_handle_list(
            _source_ids_to_indices(
                decision.get("contradiction_candidate_ids"),
                source_to_index=candidate_source_to_index,
                context="identity proposal contradiction candidates",
            ),
            alias_to_source=candidate_aliases,
            context="identity proposal contradiction candidates",
        )
    elif stage == "review":
        decision["selected_candidate_id"] = _restore_optional_handle(
            _optional_source_id_to_index(
                decision.get("selected_candidate_id"),
                source_to_index=candidate_source_to_index,
                context="identity review selected candidate",
            ),
            alias_to_source=candidate_aliases,
            context="identity review selected candidate",
        )
        decision["rejected_candidate_ids"] = _restore_handle_list(
            _source_ids_to_indices(
                decision.get("rejected_candidate_ids"),
                source_to_index=candidate_source_to_index,
                context="identity review rejected candidates",
            ),
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


def _source_ids_to_indices(
    value: object,
    *,
    source_to_index: Mapping[str, int],
    context: str,
) -> list[int]:
    """Map validated repository identifiers back to prompt-local indices."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    indices: list[int] = []
    for source_id in value:
        if not isinstance(source_id, str):
            raise ValueError(f"{context} entries must be text")
        index = source_to_index.get(source_id)
        if index is None:
            raise ValueError(f"{context} cites an unknown repository id")
        indices.append(index)
    return indices


def _optional_source_id_to_index(
    value: object,
    *,
    source_to_index: Mapping[str, int],
    context: str,
) -> int | None:
    """Map one optional repository identifier back to a prompt-local index."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be text or null")
    index = source_to_index.get(value)
    if index is None:
        raise ValueError(f"{context} cites an unknown repository id")
    return index


def _restore_handle_list(
    value: object,
    *,
    alias_to_source: Mapping[int, str],
    context: str,
) -> list[str]:
    """Translate validated prompt indices back to repository identifiers."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    source_ids: list[str] = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{context} entries must be integers")
        source_id = alias_to_source.get(index)
        if source_id is None:
            raise ValueError(f"{context} cites an unknown prompt index")
        source_ids.append(source_id)
    return source_ids


def _restore_optional_handle(
    value: object,
    *,
    alias_to_source: Mapping[int, str],
    context: str,
) -> str | None:
    """Translate one optional prompt index back to a repository identifier."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer or null")
    source_id = alias_to_source.get(value)
    if source_id is None:
        raise ValueError(f"{context} cites an unknown prompt index")
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
        violations = [
            {
                "code": "semantic_noop",
                "field": path,
                "expected": "a replacement that differs from the current value",
            }
            for path in sorted(no_op_paths)
        ]
        raise IdentityContractViolation(
            violations=violations,
            message=(
                "identity patches are no-ops: "
                f"{sorted(no_op_paths)}"
            ),
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


def _bounded_validation_descriptor(
    violations: list[dict[str, str]],
) -> str:
    """Serialize all known violations into a bounded repair descriptor."""

    complete = [
        {
            "code": entry["code"],
            "field": entry["field"],
            "expected": entry["expected"],
        }
        for entry in violations
    ]
    serialized = json.dumps(
        {"violations": complete},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) <= IDENTITY_REPAIR_DESCRIPTOR_CHAR_LIMIT:
        return serialized

    # The typed exception retains complete expected values; the retry form
    # keeps every violation's stable code and field within the strict bound.
    compact = []
    compact_field_names = {
        "action": "action",
        "candidate_index": "candidate",
        "character_authorship": "authorship",
        "character_owned_abstraction": "owned",
        "confidence": "confidence",
        "contradiction_candidate_indices": "contradictions",
        "evidence_indices": "evidence",
        "global_applicability": "global",
        "identity_relevance": "relevance",
        "private_detail_risk": "privacy",
        "proposed_changes": "changes",
    }
    for entry in violations:
        field = entry["field"]
        for prefix in (
            "identity proposal decision.",
            "identity review decision.",
        ):
            if field.startswith(prefix):
                field = field[len(prefix):]
                break
        field = compact_field_names.get(field, field)
        compact.append({
            "code": entry["code"],
            "field": field,
        })
    serialized = json.dumps(
        {"violations": compact},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) > IDENTITY_REPAIR_DESCRIPTOR_CHAR_LIMIT:
        raise ValueError("identity repair descriptor exceeds its bound")
    return serialized


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

    base_messages = [
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ]
    messages = list(base_messages)
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
            validation_error_codes.append(f"{stage}.provider_error")
            await _record_trace_attempt(
                trace_id=effective_trace_id,
                stage=stage,
                attempt_index=attempt_index,
                messages=messages,
                raw_output="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                validation_error="",
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

        validation_error = ""
        violation_codes: list[str] = []
        try:
            parsed_output = parse_llm_json_output(
                raw_output,
                expected_output_format=expected_output_format,
            )
            if not isinstance(parsed_output, Mapping):
                raise IdentityContractViolation(
                    violations=[{
                        "code": "malformed_json",
                        "field": "parsed_output",
                        "expected": "a complete closed JSON object",
                    }],
                )
            validated = validator(parsed_output)
        except IdentityContractViolation as exc:
            violation_codes = [
                f"{stage}.{entry['code']}"
                for entry in exc.violations
            ]
            validation_error = _bounded_validation_descriptor(
                exc.violations
            )
            ended_at = time.perf_counter()
            validation_error_codes.extend(violation_codes)
            await _record_trace_attempt(
                trace_id=effective_trace_id,
                stage=stage,
                attempt_index=attempt_index,
                messages=messages,
                raw_output=raw_output,
                parsed_output=parsed_output,
                parse_status="contract_error",
                status="failed",
                validation_error=validation_error,
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
            messages = [
                base_messages[0],
                base_messages[1],
                HumanMessage(content=validation_error),
            ]
            continue
        except (KeyError, TypeError, ValueError) as exc:
            violation_codes.append(f"{stage}.contract_error")
            validation_error = json.dumps(
                {
                    "violations": [{
                        "code": "malformed_json",
                        "field": "parsed_output",
                        "expected": "a complete closed JSON object",
                    }],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            ended_at = time.perf_counter()
            validation_error_codes.extend(violation_codes)
            await _record_trace_attempt(
                trace_id=effective_trace_id,
                stage=stage,
                attempt_index=attempt_index,
                messages=messages,
                raw_output=raw_output,
                parsed_output=parsed_output,
                parse_status="contract_error",
                status="failed",
                validation_error=validation_error,
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
            messages = [
                base_messages[0],
                base_messages[1],
                HumanMessage(content=validation_error),
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
            validation_error=validation_error,
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
    validation_error: str,
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
        validation_error=validation_error,
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
