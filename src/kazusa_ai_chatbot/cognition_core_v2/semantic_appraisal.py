"""Bounded semantic appraisal that emits propositions instead of state writes."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.contracts import CognitionChainServices
from kazusa_ai_chatbot.cognition_core_v2.contracts import SemanticProposition
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_stage,
)


SEMANTIC_APPRAISAL_PROMPT = '''You assess only the causal meaning of the current evidence.
Use the supplied permitted causal roots. Do not select actions, write dialogue,
or infer facts absent from the evidence.

# Generation Procedure
For each supported causal root, decide whether the evidence establishes it.
Return a false proposition only when an active causal root is established as
resolved or absent by supplied evidence. Do not treat a root as absent merely
because it is unmentioned. Return every proposition with a supplied source ref.

# Active-Cause Resolution Rule
active_causal_roots identifies causes that were active before this evidence.
When supplied evidence expressly establishes that one of those causes is
resolved, ended, gone, or no longer applies, return present=false for that
active root. Do not return present=true merely because the evidence mentions
the earlier cause or its history. In particular, return present=false for an
active prior_threat_reduction root when the evidence establishes that the
previously active reduction event is now resolved and no longer applies.

# Canonical Causal-Root Glossary
- goal_reward: a valued goal has made meaningful progress or reached reward.
- credible_threat: credible expected harm remains possible and unresolved.
- goal_obstruction: an important goal is being blocked by a meaningful obstacle.
- valued_loss: a valued person, object, relationship, opportunity, or outcome
  is actually lost or irreversibly unavailable in the current judgment.
- contamination_or_norm_rejection: contamination or a rejected norm creates
  identity-relevant aversion.
- prediction_error: an unexpected event requires attention or belief revision.
- bond_attachment: an established valued bond or desire for closeness is salient.
- observed_other_affect: evidence establishes another person's affect or need.
- attributed_benefit: another agent intentionally provided a meaningful benefit.
- rival_threat: a valued bond faces a credible rival or exclusivity threat.
- upward_comparison: another's desired advantage creates an attainable comparison.
- self_caused_achievement: the active character caused a valued achievement.
- global_standard_threat: a broad self-standard or reputation is threatened.
- self_caused_harm: the active character caused specific harm needing repair.
- minor_social_error: a visible minor social error has limited moral severity.
- valuable_knowledge_gap: an important and learnable knowledge gap is present.
- vastness: exceptional scale or complexity requires model accommodation.
- autobiographical_continuity: a memory links the present self to a personally
  meaningful past, supporting nostalgic remembrance or continuity.
- connection_gap: desired connection exceeds perceived available connection.
- prior_threat_reduction: a previously active threat has materially reduced.
- low_purpose_coherence: purpose, agency, or viable long-horizon direction is low.

# Distinguishing Nostalgia From Sadness
Use autobiographical_continuity when evidence is primarily a personally
meaningful memory connecting the present to a cherished past. A past memory,
even one that is tender or unavailable now, does not by itself establish
valued_loss. Use valued_loss only when the evidence also establishes an actual
irreversible loss or current unavailability that is central to the judgment.

# Output Format
Return a JSON object with "propositions": a list of objects. Each object has
"root_id" (string), "present" (boolean), "causal_source_ref" (string), and
"semantic_basis" (short string).
'''


async def appraise_semantic_sources(
    source_summaries: Sequence[Mapping[str, str]],
    allowed_root_ids: set[str],
    services: CognitionChainServices,
    active_root_ids: set[str] | None = None,
) -> list[SemanticProposition]:
    """Ask one bounded appraisal question and validate its typed propositions.

    Args:
        source_summaries: Prompt-safe current evidence with stable source refs.
        allowed_root_ids: Causal roots available to the local reducer.
        services: Existing V1 LLM binding and cognition configuration.
        active_root_ids: Existing causal roots that evidence may resolve.

    Returns:
        Structurally validated semantic propositions without applying state.
    """

    payload = {
        "permitted_causal_roots": sorted(allowed_root_ids),
        "active_causal_roots": sorted(active_root_ids or set()),
        "evidence": list(source_summaries),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    started_at = time.perf_counter()
    raw_output: str | None = None
    parsed: object | None = None
    try:
        response = await services.llm.ainvoke(
            [
                SystemMessage(content=SEMANTIC_APPRAISAL_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=services.cognition_config,
        )
        raw_output = response.content
        parsed = services.parse_json(raw_output)
        propositions = _validate_propositions(parsed, allowed_root_ids)
    except Exception as exc:
        ended_at = time.perf_counter()
        capture_validation_stage(
            stage_id="semantic_appraisal",
            config=services.cognition_config,
            system_prompt=SEMANTIC_APPRAISAL_PROMPT,
            human_payload=payload_text,
            raw_output=raw_output,
            parsed_output=parsed,
            parse_status="failed",
            started_at=started_at,
            ended_at=ended_at,
            error=str(exc),
        )
        raise
    ended_at = time.perf_counter()
    capture_validation_stage(
        stage_id="semantic_appraisal",
        config=services.cognition_config,
        system_prompt=SEMANTIC_APPRAISAL_PROMPT,
        human_payload=payload_text,
        raw_output=raw_output,
        parsed_output=parsed,
        parse_status="succeeded",
        started_at=started_at,
        ended_at=ended_at,
    )
    return propositions


def _validate_propositions(
    parsed: object,
    allowed_root_ids: set[str],
) -> list[SemanticProposition]:
    """Retain only structurally valid propositions inside the allowed root set."""

    if not isinstance(parsed, Mapping):
        raise ValueError("semantic appraisal must return an object")
    raw_propositions = parsed.get("propositions")
    if not isinstance(raw_propositions, list):
        raise ValueError("semantic appraisal propositions must be a list")
    propositions: list[SemanticProposition] = []
    for raw_proposition in raw_propositions:
        if not isinstance(raw_proposition, Mapping):
            raise ValueError("semantic proposition must be an object")
        root_id = raw_proposition.get("root_id")
        present = raw_proposition.get("present")
        causal_source_ref = raw_proposition.get("causal_source_ref")
        semantic_basis = raw_proposition.get("semantic_basis")
        if root_id not in allowed_root_ids:
            raise ValueError("semantic proposition uses an unsupported causal root")
        if not isinstance(present, bool):
            raise ValueError("semantic proposition presence must be boolean")
        if not isinstance(causal_source_ref, str) or not causal_source_ref:
            raise ValueError("semantic proposition requires a causal source ref")
        if not isinstance(semantic_basis, str) or not semantic_basis:
            raise ValueError("semantic proposition requires a semantic basis")
        propositions.append(
            SemanticProposition(
                root_id=root_id,
                present=present,
                causal_source_ref=causal_source_ref,
                semantic_basis=semantic_basis,
            )
        )
    return propositions
