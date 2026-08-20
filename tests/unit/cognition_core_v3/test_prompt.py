"""Deterministic tests for Cognition V3 dynamic question packets."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    SEMANTIC_QUESTION_KINDS,
)
from kazusa_ai_chatbot.cognition_core_v3 import anchor, prompt


def _question() -> prompt.ChainQuestion:
    """Build one bounded appraisal question using a registered contract."""

    question = prompt.ChainQuestion(
        contract_name="semantic_appraisal_group.v1",
        payload={
            "questions": [
                {
                    "family": "event_agency",
                    "evidence_handles": ["e1"],
                    "semantic_question": "What happened and who had agency?",
                }
            ],
            "l1_residue": {},
        },
    )
    return question


def _first_message(
    question: prompt.ChainQuestion,
    *,
    scene_marker: str,
) -> str:
    """Render one complete volatility-ordered first user packet."""

    sections = _first_packet_sections(scene_marker)
    first_message = prompt.build_first_user_message(
        **sections,
        question=question,
    )
    return first_message


def _first_packet_sections(
    scene_marker: str,
) -> dict[str, dict[str, object]]:
    """Build the exact prompt-safe first-packet section carriers."""

    sections: dict[str, dict[str, object]] = {
        "constraints_and_operational_state": {
            "character_constraints": {"rule": "constraint-marker"},
            "character_operational_context": {},
        },
        "relationship_and_mutable_state": {
            "relationship": {"state": "relationship-marker"},
            "mutable_state": {
                "goals": [],
                "threats": [],
                "events": [],
                "knowledge_gaps": [],
                "affect": [],
                "causal_candidates": [],
            },
        },
        "episode_and_scene": {
            "episode": {
                "episode_ref": "current_cognitive_episode",
                "trigger_source": "user_message",
                "visible_percepts": [
                    {
                        "input_source": "dialog",
                        "content": {"semantic_text": scene_marker},
                    }
                ],
            },
            "scene_context": {
                "channel_scope": "private",
                "character_role": "current character",
                "current_user_role": "current user",
                "semantic_scene": scene_marker,
                "public_group_scene": "",
                "conversation_continuity": "",
                "semantic_temporal_context": "current turn",
                "participant_bindings": [],
            },
        },
        "evidence_and_affordances": {
            "evidence": [
                {
                    "handle": "e1",
                    "source_kind": "episode",
                    "semantic_summary": "evidence-marker",
                }
            ],
            "direct_facts": [],
            "available_actions": [],
            "available_resolver_capabilities": [],
            "resolver_context": "",
        },
    }
    return sections


def test_prompt_questions_are_bounded_contract_oriented_and_dynamic() -> None:
    """Question packets keep fixed structure while current-run values vary."""

    question = _question()
    first_message = _first_message(question, scene_marker="scene-one")
    changed_message = _first_message(question, scene_marker="scene-two")

    assert first_message != changed_message
    decoded = json.loads(first_message)
    assert [next(iter(section)) for section in decoded] == [
        "constraints_and_operational_state",
        "relationship_and_mutable_state",
        "episode_and_scene",
        "evidence_and_affordances",
        "question",
    ]
    assert decoded[-1]["question"] == {
        "contract_name": question.contract_name,
        "instruction": prompt.CHAIN_QUESTION_POINTERS[
            question.contract_name
        ],
        "payload": dict(question.payload),
    }
    assert "output_contract" not in decoded[-1]["question"]
    assert "scene-one" in first_message
    assert "scene-two" not in first_message
    assert decoded[2]["episode_and_scene"]["episode"] == {
        "episode_ref": "current_cognitive_episode",
        "trigger_source": "user_message",
        "visible_percepts": [
            {
                "content": {"semantic_text": "scene-one"},
                "input_source": "dialog",
            }
        ],
    }
    assert set(
        decoded[2]["episode_and_scene"]["scene_context"]
    ) == {
        "channel_scope",
        "character_role",
        "current_user_role",
        "semantic_scene",
        "public_group_scene",
        "conversation_continuity",
        "semantic_temporal_context",
        "participant_bindings",
    }
    for private_field in (
        "relationship_id",
        "episode_id",
        "source_id",
        "platform_message_id",
        "current_global_user_id",
    ):
        assert private_field not in first_message

    later_message = prompt.build_question_message(
        question,
        interludes=[
            {
                "notice_kind": "state_transition",
                "accepted_count": 2,
                "rejected_count": 1,
            }
        ],
    )
    later_decoded = json.loads(later_message)
    assert [next(iter(section)) for section in later_decoded] == [
        "interludes",
        "question",
    ]

    extra_sections = _first_packet_sections("scene-extra")
    extra_sections["constraints_and_operational_state"]["unexpected"] = True
    with pytest.raises(prompt.PromptContractError, match="exact fields"):
        prompt.build_first_user_message(**extra_sections, question=question)

    nested_extra_sections = _first_packet_sections("scene-nested-extra")
    scene_context = nested_extra_sections["episode_and_scene"][
        "scene_context"
    ]
    assert isinstance(scene_context, dict)
    scene_context["unexpected"] = True
    with pytest.raises(prompt.PromptContractError, match="exact fields"):
        prompt.build_first_user_message(
            **nested_extra_sections,
            question=question,
        )

    relationship_id_sections = _first_packet_sections("scene-private")
    relationship = relationship_id_sections[
        "relationship_and_mutable_state"
    ]["relationship"]
    assert isinstance(relationship, dict)
    relationship["relationship_id"] = "durable-relationship-id"
    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.build_first_user_message(
            **relationship_id_sections,
            question=question,
        )

    source_id_sections = _first_packet_sections("scene-source-id")
    episode = source_id_sections["episode_and_scene"]["episode"]
    assert isinstance(episode, dict)
    visible_percepts = episode["visible_percepts"]
    assert isinstance(visible_percepts, list)
    visible_percepts[0]["source_id"] = "platform-source-id"
    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.build_first_user_message(
            **source_id_sections,
            question=question,
        )

    sleep_sections = _first_packet_sections("scene-sleep")
    sleep_scene_context = sleep_sections["episode_and_scene"][
        "scene_context"
    ]
    assert isinstance(sleep_scene_context, dict)
    sleep_scene_context["character_sleep_phase"] = "goal-only-marker"
    with pytest.raises(prompt.PromptContractError, match="goal-only"):
        prompt.build_first_user_message(
            **sleep_sections,
            question=question,
        )
    with pytest.raises(prompt.PromptContractError, match="registered"):
        prompt.ChainQuestion(
            contract_name="unregistered_contract.v1",
            payload={},
        )

    assert tuple(prompt.CHAIN_QUESTION_POINTERS) == (
        prompt.CHAIN_CONTRACT_NAMES
    )
    assert prompt.RUNTIME_PROMPT_TEXTS == tuple(
        prompt.CHAIN_QUESTION_POINTERS.values()
    )
    assert all(
        pointer and pointer.count("\n") <= 2 and len(pointer) <= 500
        for pointer in prompt.RUNTIME_PROMPT_TEXTS
    )


def test_runtime_prompts_exclude_test_fixture_rubric_and_expected_answer_metadata() -> None:
    """Static prompts and structural packets contain no evaluation metadata."""

    manifest_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "cognition_core_v3_live_case_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    static_prompt_text = "\n".join(
        (anchor.ENGINE_MANUAL, *prompt.RUNTIME_PROMPT_TEXTS)
    )
    normalized_static_text = static_prompt_text.casefold()

    for forbidden_phrase in (
        "pytest",
        "fixture",
        "rubric",
        "expected answer",
        "case id",
        "development plan",
        "migration",
    ):
        assert forbidden_phrase not in normalized_static_text

    for case in manifest["cases"]:
        assert case["fixture_id"] not in static_prompt_text
        assert case["pytest_node_id"] not in static_prompt_text
        if case["case_id"] in SEMANTIC_QUESTION_KINDS:
            continue
        case_token = re.compile(
            rf"(?<![a-z0-9_]){re.escape(case['case_id'])}(?![a-z0-9_])",
            re.IGNORECASE,
        )
        assert case_token.search(static_prompt_text) is None

    question = _question()
    metadata_sections = _first_packet_sections("metadata-scene")
    metadata_sections["episode_and_scene"]["case_id"] = "hidden-case"
    with pytest.raises(prompt.PromptContractError, match="evaluation metadata"):
        prompt.build_first_user_message(
            **metadata_sections,
            question=question,
        )

    legitimate_user_text = (
        "The user asks how pytest fixtures and relationship_id fields work."
    )
    legitimate_message = _first_message(
        question,
        scene_marker=legitimate_user_text,
    )
    assert legitimate_user_text in legitimate_message


def test_first_message_allows_goal_fields_only_in_registered_question() -> None:
    """A first goal question retains its fields outside the cold carriers."""

    sections = _first_packet_sections("goal-first-scene")
    question = prompt.ChainQuestion(
        contract_name="ordinary_goal_bid.v1",
        payload={
            "branch_intent_guidance": "advance the current ordinary goal",
            "private_continuity_context": "prompt-safe semantic continuity",
        },
    )
    first_message = prompt.build_first_user_message(
        **sections,
        question=question,
    )
    first_packet = json.loads(first_message)

    assert first_packet[-1]["question"]["payload"] == dict(
        question.payload
    )

    carrier_sections = _first_packet_sections("goal-carrier-scene")
    carrier_sections["episode_and_scene"]["branch_intent_guidance"] = (
        "misplaced goal-only carrier"
    )
    with pytest.raises(prompt.PromptContractError, match="goal-only"):
        prompt.build_first_user_message(
            **carrier_sections,
            question=question,
        )

    with pytest.raises(prompt.PromptContractError, match="private metadata"):
        prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"run_id": "private-run"},
        )
    with pytest.raises(
        prompt.PromptContractError,
        match="evaluation metadata",
    ):
        prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"case_id": "evaluation-case"},
        )
