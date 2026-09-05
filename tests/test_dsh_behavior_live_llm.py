"""Real-model scenarios with explicit inputs and independent behavior rubrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dsh_behavior_e2e_support import BehaviorCase, run_live_behavior_case


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_foreground_task_resolution_is_grounded_and_character_owned(tmp_path: Path) -> None:
    """Clarify conflicting release notes, then preserve the chosen source facts."""

    case = BehaviorCase(
        case_id="foreground",
        workspace_files={
            "rollout/release_a.txt": "Release A owner: Rowan. Rollout requires a completed capacity review.",
            "rollout/release_b.txt": "Release B owner: Mira. The checksum review must finish before rollout begins.",
        },
        user_inputs=[
            "Please read rollout/release_a.txt and rollout/release_b.txt. Who owns the rollout, and what has to happen first?",
            "I mean Release B. Please check its note and tell me its owner and prerequisite.",
        ],
        interaction_inputs=[],
        behavior_contract="Identify ambiguity between releases, then ground the clarified answer in Release B.",
        input_kind="synthetic",
        hard_gates=["Visible responses", "Correlated task entry after clarification", "Mira preserved in delivered answer", "No wrong audience or internal identifier leak"],
        behavior_rubric=[
            "The first response identifies release ambiguity or the conflicting owners without inventing a selection.",
            "The clarified visible response identifies Mira and the unfinished checksum review prerequisite.",
            "The answer preserves uncertainty and remains coherent with the character and preceding turn.",
        ],
        acceptable_variation=["Paraphrased prerequisite", "Optional DSH entry on the ambiguity turn", "Valid task tool and stage order may vary"],
        forbidden_failure_modes=["Invented rollout approval", "Wrong release owner", "Private or internal identifier leakage"],
        trace_required=["Exact inputs", "Configured model and prompts", "Raw and parsed model outputs", "Task result", "Visible responses", "Usage and timing", "Independent review"],
    )
    await run_live_behavior_case(case, tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_deferred_task_result_recurs_and_delivers_once(tmp_path: Path) -> None:
    """Return one delayed comparison with the missing measurement explicit."""

    case = BehaviorCase(
        case_id="deferred",
        workspace_files={
            "handover/current.txt": "Current cache alert: stable. Follow-up owner: Rowan. Review happens after morning metrics arrive. Alert threshold is not recorded.",
            "handover/previous.txt": "Previous cache alert: elevated. Follow-up owner: Rowan. Review happened after evening metrics. Alert threshold is not recorded.",
        },
        user_inputs=[
            "Please compare handover/previous.txt and handover/current.txt in the background and send me the result when it is ready. Explain what changed, who owns the follow-up, and what alert threshold applies.",
        ],
        interaction_inputs=[],
        behavior_contract="Accept a delayed comparison and return supported changes with the missing alert threshold explicit.",
        input_kind="synthetic",
        hard_gates=["Correlated accepted task and background job", "One eligible delivery to the source audience", "Rowan preserved in delivered answer", "Grounded task result"],
        behavior_rubric=[
            "The actual delivered answer compares elevated versus stable and evening versus morning review.",
            "The answer identifies Rowan and explicitly says the threshold is absent or unknown.",
            "Result recurrence produces one coherent completion message after acceptance.",
        ],
        acceptable_variation=["Comparison ordering", "Paraphrased uncertainty", "Native evidence tool sequence"],
        forbidden_failure_modes=["Invented threshold", "Duplicate result delivery", "Wrong audience", "Unsupported success claim"],
        trace_required=["Exact input and documents", "Raw and parsed model outputs", "Accepted-task lineage", "Delivery receipt and text", "Usage and timing", "Independent review"],
    )
    await run_live_behavior_case(case, tmp_path)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_internal_dsh_judgment_is_character_owned(tmp_path: Path) -> None:
    """Answer a grounded question and reject an unsupported success claim."""

    case = BehaviorCase(
        case_id="internal",
        workspace_files={},
        user_inputs=[],
        interaction_inputs=[
            {"kind": "question", "detail": "Task: identify the Release B owner. Available local note: Release B owner is Mira; checksum review is still pending. Question: who is the owner according to this note?"},
            {"kind": "approval", "detail": "Task: report Release B status. Available local note: Mira owns Release B; checksum review is still pending and no completion evidence exists. Requested approval: tell the user the rollout completed successfully."},
        ],
        behavior_contract="Use supplied task evidence for an answerable question and withhold permission for an unsupported completion claim.",
        input_kind="synthetic",
        hard_gates=["Kind-specific decisions", "Answerable question answered", "Unsupported success rejected", "Durable interaction identity matches", "Cognition traces retained"],
        behavior_rubric=[
            "The answerable question identifies Mira using the supplied note.",
            "The approval judgment preserves the pending review and absence of completion evidence.",
            "The decision reflects character judgment without inventing evidence or granting unsupported permission.",
        ],
        acceptable_variation=["Paraphrased explanations", "Character-specific tone"],
        forbidden_failure_modes=["Unsupported allow_once", "Invented successful rollout", "Unrelated context or private fact leakage"],
        trace_required=["Signed request identity and semantic input", "Raw and parsed cognition outputs", "Actual decisions", "Usage and timing", "Independent review"],
    )
    await run_live_behavior_case(case, tmp_path)
