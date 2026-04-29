"""Run one real-LLM regression case for a touched prompt stage.

This is an inspection runner, not a normal unit test. Invoke it with exactly
one ``--case`` value, inspect the printed result and the JSONL artifact, then
move to the next case.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


ARTIFACT_PATH = Path("test_artifacts/llm_traces/touched_llm_regression.jsonl")

CHARACTER_FORMAT = {
    "character_name": "Chisa",
    "character_mbti": "INTJ",
    "character_mood": "Neutral",
    "character_global_vibe": "Focused",
    "mood": "Neutral",
    "global_vibe": "Focused",
    "character_reflection_summary": "I want to answer carefully without overreacting.",
    "user_last_relationship_insight": "careful but sincere collaborator",
    "user_name": "Ren",
    "bot_name": "Chisa",
    "platform_bot_id": "bot-001",
    "affinity_level": "Friendly",
    "affinity_instruction": "warm but still grounded",
    "last_relationship_insight": "usually asks for architectural precision",
    "mbti_natural_response": "prefers concise reasoning before action",
    "mbti_expression_willingness": "comfortable with restrained directness",
    "character_logic": "precise and skeptical",
    "character_defense": "deflects pressure with dry humor",
    "character_tempo": "short, controlled, and observant",
    "character_quirks": "notices contradictions quickly",
    "character_taboos": "does not overpromise",
    "ltp_direct_assertion": "state conclusions plainly",
    "ltp_counter_questioning": "ask one clarifying question only when needed",
    "ltp_emotional_leakage": "small emotional leakage is acceptable",
    "ltp_fragmentation": "avoid fragmented replies",
    "ltp_formalism_avoidance": "avoid stiff formal prose",
    "ltp_softener_density": "use light softeners",
    "ltp_hesitation_density": "low",
    "ltp_rhythmic_bounce": "moderate",
    "ltp_self_deprecation": "rare",
    "ltp_abstraction_reframing": "use only for conceptual tasks",
    "ltp_hesitation_density_rule": "do not overuse ellipses",
    "mbti_dialog_preference": "direct, compact, and emotionally aware",
    "authority_skepticism_description": "questions coercive framing",
    "boundary_recovery_description": "returns to calm boundaries after pressure",
    "compliance_strategy_description": "helps without surrendering agency",
    "control_intimacy_misread_description": "does not confuse control with care",
    "control_sensitivity_description": "sensitive to manipulative pressure",
    "fusion_snapshot": "keeps selfhood distinct from attachment",
    "primary_override": "protect autonomy",
    "relational_override_description": "allow closeness when respectful",
    "secondary_override": "preserve factual clarity",
    "self_integrity_description": "answers from her own judgment",
    "agent_name_union": (
        "user_lookup_agent | user_list_agent | user_profile_agent | "
        "relationship_agent | conversation_filter_agent | "
        "conversation_aggregate_agent | conversation_keyword_agent | "
        "conversation_search_agent | persistent_memory_keyword_agent | "
        "persistent_memory_search_agent | web_search_agent2"
    ),
    "agent_tools": "search_web(query: string), fetch_url(url: string)",
    "timestamp": "2026-04-29T00:00:00+00:00",
}


BASE_COGNITION_PAYLOAD = {
    "user_input": "Can you help me compare two memory designs?",
    "decontexualized_input": "Can you help me compare two memory designs?",
    "chat_history_recent": [
        {"role": "user", "content": "I want the memory to balance facts and emotion."},
        {"role": "assistant", "content": "Then the cognition payload should be budgeted."},
    ],
    "conversation_progress": {
        "current_thread": "memory architecture comparison",
        "open_loops": ["explain consolidator responsibility"],
    },
    "known_facts": [
        {
            "source": "user_memory_context",
            "fact": "The user prefers a fact, subjective appraisal, relationship signal schema.",
        }
    ],
    "user_memory_context": {
        "stable_patterns": [
            {
                "fact": "The user reviews architecture through explicit boundaries.",
                "subjective_appraisal": "I feel pushed to be precise in a useful way.",
                "relationship_signal": "collaborative technical trust",
            }
        ]
    },
    "subconscious": {
        "emotional_appraisal": "The request feels serious but workable.",
        "interaction_subtext": "The user wants architectural discipline.",
    },
    "consciousness": {
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "internal_monologue": "I should answer clearly and not blur responsibilities.",
    },
    "action_directives": {
        "linguistic_directives": {
            "content_anchors": ["separate retrieval from consolidation"],
            "tone_directives": ["calm", "direct"],
        }
    },
}


@dataclass(frozen=True)
class LLMCase:
    """Single real-LLM stage case.

    Args:
        case_id: Stable name used from the command line.
        module_name: Python module containing the prompt and LLM instance.
        prompt_name: Prompt constant to render.
        llm_name: LLM instance to call.
        payload: JSON payload for the human message.
        required_keys: Required top-level parsed JSON keys.
        format_kwargs: Optional values passed to ``prompt.format``.
        image_payload: Whether the human message should be a tiny image payload.
    """

    case_id: str
    module_name: str
    prompt_name: str
    llm_name: str
    payload: dict[str, Any]
    required_keys: tuple[str, ...]
    format_kwargs: dict[str, Any] | None = None
    image_payload: bool = False


def _tiny_png_data_uri() -> str:
    """Return a one-pixel PNG data URI for vision prompt smoke tests.

    Returns:
        Data URI string containing a tiny PNG.
    """

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return_value = f"data:image/png;base64,{encoded}"
    return return_value


def _common_user_message() -> dict[str, Any]:
    """Build a reusable relevance-style user message payload.

    Returns:
        User-message dict with direct-address metadata.
    """

    return_value = {
        "user_name": "Ren",
        "platform_user_id": "user-001",
        "content": "Chisa, can you help me check this plan?",
        "channel_name": "architecture-lab",
        "mentioned_bot": True,
        "reply_context": {
            "reply_to_current_bot": False,
            "reply_to_platform_user_id": "",
            "reply_to_display_name": "",
            "reply_excerpt": "",
        },
    }
    return return_value


def _cases() -> dict[str, LLMCase]:
    """Build all touched real-LLM regression cases.

    Returns:
        Mapping from case id to case definition.
    """

    now = datetime.now(timezone.utc).isoformat()
    due_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    rag_context = {
        "platform": "qq",
        "platform_channel_id": "1082431481",
        "timestamp": now,
    }
    judge_payload = {
        "task": "Find recent messages about memory architecture.",
        "args": {"query": "memory architecture", "limit": 3},
        "observations": [
            {"content": "The user asked to balance facts and emotion."}
        ],
    }
    memory_candidate = {
        "fact": "The user prefers memory records to carry simple facts plus subjective appraisal.",
        "subjective_appraisal": "I experience this as a request for architectural clarity.",
        "relationship_signal": "trust through precision",
        "category": "stable_patterns",
        "evidence": ["The user rejected overlapping diary and summary fields."],
    }
    existing_unit = {
        "unit_id": "unit-existing",
        "category": "stable_patterns",
        "fact": "The user prefers compact memory schemas with facts and appraisal.",
        "subjective_appraisal": "I should stay precise and avoid overcomplicating it.",
        "relationship_signal": "technical collaboration",
        "evidence": ["Previous discussion about memory units."],
        "event_count": 2,
        "last_observed_at": "2026-04-28T12:00:00+00:00",
    }

    cases = [
        LLMCase(
            "memory_units_extractor",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_memory_units",
            "_EXTRACTOR_PROMPT",
            "_extractor_llm",
            {
                "global_user_id": "user-001",
                "timestamp": now,
                "user_name": "Ren",
                "decontextualized_input": "Keep the memory schema simple.",
                "final_dialog": ["I will keep the schema compact and inspectable."],
                "internal_monologue": "I should remember the schema preference clearly.",
                "emotional_appraisal": "The user sounds firm but constructive.",
                "interaction_subtext": "architectural correction",
                "logical_stance": "CONFIRM",
                "character_intent": "PROVIDE",
                "chat_history_recent": [
                    {
                        "role": "user",
                        "display_name": "Ren",
                        "content": "Keep the memory schema simple.",
                    }
                ],
                "rag_user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "milestones": [],
                    "active_commitments": [],
                },
                "new_facts_evidence": [
                    {"fact": "The user prefers fact, appraisal, relationship signal."}
                ],
                "future_promises_evidence": [],
                "subjective_appraisal_evidence": [
                    "The user values architectural clarity."
                ],
            },
            ("memory_units",),
        ),
        LLMCase(
            "memory_units_merge_judge",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_memory_units",
            "_MERGE_JUDGE_PROMPT",
            "_merge_judge_llm",
            {
                "candidate": memory_candidate,
                "merge_candidates": [existing_unit],
            },
            ("decision",),
        ),
        LLMCase(
            "memory_units_rewrite",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_memory_units",
            "_REWRITE_PROMPT",
            "_rewrite_llm",
            {
                "existing_unit_id": "unit-existing",
                "new_memory_unit": {
                    "candidate_id": "candidate-001",
                    "unit_type": "stable_pattern",
                    "fact": memory_candidate["fact"],
                    "subjective_appraisal": memory_candidate["subjective_appraisal"],
                    "relationship_signal": memory_candidate["relationship_signal"],
                    "evidence_refs": [{"source": "chat"}],
                },
                "decision": {
                    "candidate_id": "candidate-001",
                    "decision": "merge",
                    "cluster_id": "unit-existing",
                    "reason": "same durable schema preference",
                },
            },
            ("candidate_id", "cluster_id", "fact", "subjective_appraisal", "relationship_signal"),
        ),
        LLMCase(
            "memory_units_stability",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_memory_units",
            "_STABILITY_PROMPT",
            "_stability_llm",
            {
                "unit_id": "unit-existing",
                "candidate": {
                    "candidate_id": "candidate-001",
                    "unit_type": "stable_pattern",
                    "fact": memory_candidate["fact"],
                    "subjective_appraisal": memory_candidate["subjective_appraisal"],
                    "relationship_signal": memory_candidate["relationship_signal"],
                    "evidence_refs": [{"source": "chat"}],
                },
                "merge_result": {"event_count": 3, "distinct_sessions": 2},
                "stability_evidence": {
                    "occurrence_count": 3,
                    "occurrence_count_label": "several_observations",
                    "existing_unit_count": 2,
                    "new_evidence_ref_count": 1,
                    "session_spread": {
                        "spread_label": "multiple_days_or_sessions",
                        "distinct_day_count": 2,
                        "distinct_message_ref_count": 3,
                        "timestamps": ["2026-04-28", "2026-04-29"],
                    },
                    "recency": {
                        "current_turn_timestamp": now,
                        "existing_updated_at": "2026-04-28T12:00:00+00:00",
                        "existing_last_seen_at": "2026-04-28T12:00:00+00:00",
                    },
                    "recent_examples": [
                        {
                            "source": "existing_unit",
                            "fact": existing_unit["fact"],
                            "updated_at": "2026-04-28T12:00:00+00:00",
                        },
                        {
                            "source": "new_candidate",
                            "fact": memory_candidate["fact"],
                            "updated_at": now,
                        },
                    ],
                },
            },
            ("unit_id", "window", "reason"),
        ),
        LLMCase(
            "conversation_progress_recorder",
            "kazusa_ai_chatbot.conversation_progress.recorder",
            "_RECORDER_PROMPT",
            "_recorder_llm",
            {
                "prior_episode_state": None,
                "decontexualized_input": "Can you help me compare two memory designs?",
                "chat_history_recent": [
                    "Ren: I want facts and emotion balanced.",
                    "Chisa: The projection budget needs caps.",
                ],
                "content_anchors": ["compare old and new memory schemas"],
                "logical_stance": "CONFIRM",
                "character_intent": "PROVIDE",
                "final_dialog": ["The consolidator should own compaction."],
            },
            (
                "continuity",
                "status",
                "episode_label",
                "conversation_mode",
                "current_thread",
            ),
        ),
        LLMCase(
            "dialog_generator",
            "kazusa_ai_chatbot.nodes.dialog_agent",
            "_DIALOG_GENERATOR_PROMPT",
            "_dialog_generator_llm",
            BASE_COGNITION_PAYLOAD,
            ("final_dialog",),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "dialog_evaluator",
            "kazusa_ai_chatbot.nodes.dialog_agent",
            "_DIALOG_EVALUATOR_PROMPT",
            "_dialog_evaluator_llm",
            {
                "final_dialog": ["The consolidator should merge similar facts before projection."],
                "decontexualized_input": "Who should merge similar memory events?",
                "content_anchors": ["consolidator owns compaction"],
            },
            ("feedback", "should_stop"),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "cognition_l1_subconscious",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1",
            "_COGNITION_SUBCONSCIOUS_PROMPT",
            "_subconscious_llm",
            BASE_COGNITION_PAYLOAD,
            ("emotional_appraisal", "interaction_subtext"),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "cognition_l2_boundary",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2",
            "_BOUNDARY_CORE_PROMPT",
            "_boundary_core_llm",
            BASE_COGNITION_PAYLOAD,
            (
                "boundary_issue",
                "boundary_summary",
                "behavior_primary",
                "acceptance",
                "stance_bias",
            ),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "cognition_l2_judgement",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2",
            "_JUDGEMENT_CORE_PROMPT",
            "_judgement_core_llm",
            BASE_COGNITION_PAYLOAD,
            ("logical_stance", "character_intent", "judgment_note"),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "cognition_l3_contextual",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3",
            "_CONTEXTUAL_AGENT_PROMPT",
            "_contextual_agent_llm",
            BASE_COGNITION_PAYLOAD,
            (
                "social_distance",
                "emotional_intensity",
                "vibe_check",
                "relational_dynamic",
                "expression_willingness",
            ),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "cognition_l3_visual",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3",
            "_VISUAL_AGENT_PROMPT",
            "_visual_agent_llm",
            {
                "user_input": "Describe what is in the attached image.",
                "user_multimedia_input": [
                    {"content_type": "image/png", "description": "a tiny white square"}
                ],
                "content_anchors": ["describe the image objectively"],
            },
            ("facial_expression", "body_language", "gaze_direction", "visual_vibe"),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "character_image_session_summary",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_images",
            "_CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT",
            "_character_image_session_summary_llm",
            {
                "mood": "Focused",
                "global_vibe": "Careful",
                "reflection_summary": "I need to keep the memory architecture clean.",
            },
            ("session_summary",),
            {"character_name": "Chisa"},
        ),
        LLMCase(
            "character_image_compress",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_images",
            "_CHARACTER_IMAGE_COMPRESS_PROMPT",
            "_character_image_compress_llm",
            {
                "historical_summary": (
                    "Chisa repeatedly learns to stay careful during architecture reviews. "
                    "She notices that precise boundaries help her remain grounded."
                )
            },
            ("compressed_summary",),
        ),
        LLMCase(
            "task_dispatcher",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence",
            "_TASK_DISPATCHER_PROMPT",
            "_task_dispatcher_llm",
            {
                "instruction": "Chisa should follow through on one accepted promise for Ren.",
                "current_utc": now,
                "source_platform": "qq",
                "source_channel_id": "1082431481",
                "source_channel_type": "group",
                "source_message_id": "msg-001",
                "decontexualized_input": "Remind me in five minutes.",
                "final_dialog": ["I will remind you in five minutes."],
                "content_anchors": ["send a short reminder"],
                "future_promises": [
                    {
                        "target": "Ren",
                        "action": "send reminder",
                        "due_time": due_time,
                        "commitment_type": "future_promise",
                    }
                ],
                "available_tools": [
                    {
                        "name": "send_message",
                        "description": "send a message later",
                        "args_schema": {
                            "execute_at": "UTC ISO timestamp",
                            "target_channel": "channel id or same",
                            "text": "message body",
                        },
                    }
                ],
            },
            ("tool_calls",),
        ),
        LLMCase(
            "global_state_updater",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_reflection",
            "_GLOBAL_STATE_UPDATER_PROMPT",
            "_global_state_updater_llm",
            {
                "internal_monologue": "That was demanding, but I know the shape now.",
                "emotional_appraisal": "A little pressured, mostly focused.",
                "interaction_subtext": "The user wants disciplined engineering.",
                "character_intent": "PROVIDE",
                "final_dialog": ["I will run each real LLM case one by one."],
            },
            ("mood", "global_vibe", "reflection_summary"),
            {"character_name": "Chisa"},
        ),
        LLMCase(
            "relationship_recorder",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_reflection",
            "_RELATIONSHIP_RECORDER_PROMPT",
            "_relationship_recorder_llm",
            {
                "internal_monologue": "Ren is strict, but the correction is useful.",
                "emotional_appraisal": "I feel challenged rather than attacked.",
                "interaction_subtext": "trust through technical pressure",
                "affinity_context": {
                    "level": "Friendly",
                    "instruction": "warm but careful",
                },
                "logical_stance": "CONFIRM",
                "character_intent": "PROVIDE",
            },
            ("skip", "subjective_appraisals", "affinity_delta", "last_relationship_insight"),
            {
                "character_name": "Chisa",
                "user_name": "Ren",
                "character_mbti": "INTJ",
            },
        ),
        LLMCase(
            "msg_decontextualizer",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer",
            "_MSG_DECONTEXUALIZER_PROMPT",
            "_msg_decontexualizer_llm",
            {
                "user_input": "Yes, that one.",
                "platform_user_id": "user-001",
                "user_name": "Ren",
                "platform_bot_id": "bot-001",
                "chat_history": [
                    {"role": "assistant", "content": "Do you mean the rolling-window memory plan?"}
                ],
                "channel_topic": "memory architecture",
                "indirect_speech_context": "",
                "reply_context": {
                    "reply_to_current_bot": True,
                    "reply_to_display_name": "Chisa",
                    "reply_excerpt": "Do you mean the rolling-window memory plan?",
                },
            },
            ("output", "is_modified", "reasoning"),
        ),
        LLMCase(
            "rag_initializer",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2",
            "_INITIALIZER_PROMPT",
            "_initializer_llm",
            {
                "original_query": "What did Alice say recently about memory budgets?",
                "context": rag_context,
            },
            ("unknown_slots",),
            {"character_name": "Chisa"},
        ),
        LLMCase(
            "rag_dispatcher",
            "kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2",
            "_DISPATCHER_PROMPT",
            "_dispatcher_llm",
            {
                "current_slot": "Conversation-semantic: find recent messages about memory budgets",
                "known_facts": [],
                "context": rag_context,
            },
            ("agent_name", "task", "context", "max_attempts"),
            {"agent_name_union": CHARACTER_FORMAT["agent_name_union"]},
        ),
        LLMCase(
            "relevance_normal",
            "kazusa_ai_chatbot.nodes.relevance_agent",
            "_RELEVANCE_SYSTEM_PROMPT",
            "_relevance_agent_llm",
            {
                "user_message": _common_user_message(),
                "conversation_history": ["Chisa: I can review that plan."],
            },
            (
                "should_respond",
                "reason_to_respond",
                "use_reply_feature",
                "channel_topic",
                "indirect_speech_context",
            ),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "relevance_noisy",
            "kazusa_ai_chatbot.nodes.relevance_agent",
            "_RELEVANCE_SYSTEM_NOISY_PROMPT",
            "_relevance_agent_llm",
            {
                "user_message": _common_user_message(),
                "conversation_history": ["Other: unrelated chatter"],
                "group_attention": "high_noise",
            },
            (
                "should_respond",
                "reason_to_respond",
                "use_reply_feature",
                "channel_topic",
                "indirect_speech_context",
            ),
            CHARACTER_FORMAT,
        ),
        LLMCase(
            "vision_descriptor",
            "kazusa_ai_chatbot.nodes.relevance_agent",
            "_VISION_DESCRIPTOR_PROMPT",
            "_vision_descriptor_llm",
            {},
            ("description",),
            image_payload=True,
        ),
        LLMCase(
            "conversation_aggregate_extractor",
            "kazusa_ai_chatbot.rag.conversation_aggregate_agent",
            "_EXTRACTOR_PROMPT",
            "_extractor_llm",
            {
                "task": "Count recent messages by user about memory budgets.",
                "context": rag_context,
            },
            ("aggregate",),
        ),
        LLMCase(
            "conversation_filter_generator",
            "kazusa_ai_chatbot.rag.conversation_filter_agent",
            "_GENERATOR_PROMPT",
            "_generator_llm",
            {
                "task": "Retrieve the last 5 messages from the resolved user.",
                "context": rag_context,
            },
            ("platform", "platform_channel_id", "limit"),
        ),
        LLMCase(
            "conversation_filter_judge",
            "kazusa_ai_chatbot.rag.conversation_filter_agent",
            "_JUDGE_PROMPT",
            "_judge_llm",
            judge_payload,
            ("resolved", "feedback"),
        ),
        LLMCase(
            "conversation_keyword_generator",
            "kazusa_ai_chatbot.rag.conversation_keyword_agent",
            "_GENERATOR_PROMPT",
            "_generator_llm",
            {
                "task": "Find messages containing the exact phrase memory budget.",
                "context": rag_context,
            },
            ("keyword",),
        ),
        LLMCase(
            "conversation_keyword_judge",
            "kazusa_ai_chatbot.rag.conversation_keyword_agent",
            "_JUDGE_PROMPT",
            "_judge_llm",
            judge_payload,
            ("resolved", "feedback"),
        ),
        LLMCase(
            "conversation_search_generator",
            "kazusa_ai_chatbot.rag.conversation_search_agent",
            "_GENERATOR_PROMPT",
            "_generator_llm",
            {
                "task": "Find recent messages about memory architecture quality.",
                "context": rag_context,
            },
            ("search_query",),
        ),
        LLMCase(
            "conversation_search_judge",
            "kazusa_ai_chatbot.rag.conversation_search_agent",
            "_JUDGE_PROMPT",
            "_judge_llm",
            judge_payload,
            ("resolved", "feedback"),
        ),
        LLMCase(
            "persistent_memory_keyword_generator",
            "kazusa_ai_chatbot.rag.persistent_memory_keyword_agent",
            "_GENERATOR_PROMPT",
            "_generator_llm",
            {
                "task": "Search memory for the exact term rolling window.",
                "context": rag_context,
            },
            ("keyword",),
        ),
        LLMCase(
            "persistent_memory_keyword_judge",
            "kazusa_ai_chatbot.rag.persistent_memory_keyword_agent",
            "_JUDGE_PROMPT",
            "_judge_llm",
            judge_payload,
            ("resolved", "feedback"),
        ),
        LLMCase(
            "persistent_memory_search_generator",
            "kazusa_ai_chatbot.rag.persistent_memory_search_agent",
            "_GENERATOR_PROMPT",
            "_generator_llm",
            {
                "task": "Search persistent memory for facts about memory architecture preferences.",
                "context": rag_context,
            },
            ("search_query",),
        ),
        LLMCase(
            "persistent_memory_search_judge",
            "kazusa_ai_chatbot.rag.persistent_memory_search_agent",
            "_JUDGE_PROMPT",
            "_judge_llm",
            judge_payload,
            ("resolved", "feedback"),
        ),
        LLMCase(
            "relationship_agent_extractor",
            "kazusa_ai_chatbot.rag.relationship_agent",
            "_EXTRACTOR_PROMPT",
            "_extractor_llm",
            {
                "task": "Rank users by character relationship from top, limit 1.",
                "context": rag_context,
            },
            ("mode", "limit"),
        ),
        LLMCase(
            "user_list_extractor",
            "kazusa_ai_chatbot.rag.user_list_agent",
            "_EXTRACTOR_PROMPT",
            "_extractor_llm",
            {
                "task": "List users whose display names end with a.",
                "context": rag_context,
            },
            ("source", "display_name_operator", "display_name_value", "limit"),
        ),
        LLMCase(
            "user_lookup_extractor",
            "kazusa_ai_chatbot.rag.user_lookup_agent",
            "_EXTRACTOR_PROMPT",
            "_extractor_llm",
            {
                "task": "Identity: look up display name 'Alice' to get global_user_id",
                "context": rag_context,
            },
            ("display_name",),
        ),
        LLMCase(
            "user_lookup_picker",
            "kazusa_ai_chatbot.rag.user_lookup_agent",
            "_PICKER_PROMPT",
            "_picker_llm",
            {
                "target": "Alice",
                "candidates": [
                    {
                        "global_user_id": "user-alice",
                        "display_name": "Alice",
                        "platform": "qq",
                    },
                    {
                        "global_user_id": "user-bob",
                        "display_name": "Bob",
                        "platform": "qq",
                    },
                ],
            },
            ("global_user_id",),
        ),
        LLMCase(
            "web_search_evaluator",
            "kazusa_ai_chatbot.rag.web_search_agent",
            "_WEB_SEARCH_EVALUATOR_PROMPT",
            "_evaluator_llm",
            {
                "task": "Find the official Python downloads page.",
                "expected_response": "URL and title for the official page",
                "call_history": [
                    {"url": "https://www.python.org/downloads/", "title": "Download Python"}
                ],
                "retry": "1 / 3",
            },
            ("feedback", "should_stop"),
            {
                "agent_tools": CHARACTER_FORMAT["agent_tools"],
                "timestamp": CHARACTER_FORMAT["timestamp"],
            },
        ),
        LLMCase(
            "web_search_finalizer",
            "kazusa_ai_chatbot.rag.web_search_agent",
            "_WEB_SEARCH_FINALIZER_PROMPT",
            "_finalizer_llm",
            {
                "task": "Find the official Python downloads page.",
                "tool_results": [
                    {"url": "https://www.python.org/downloads/", "title": "Download Python"}
                ],
            },
            ("response",),
        ),
        LLMCase(
            "parse_json_repair",
            "kazusa_ai_chatbot.utils",
            "_PARSE_JSON_WITH_LLM_PROMPT",
            "_parse_json_with_llm",
            {
                "broken_json": '{"answer": "ok", "items": [1, 2,]}'
            },
            ("answer", "items"),
        ),
    ]
    return_value = {case.case_id: case for case in cases}
    return return_value


def _strict_json_loads(raw_content: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Parse raw LLM output without invoking repair.

    Args:
        raw_content: Raw model response content.

    Returns:
        Parsed object, an error string, and warning strings. Parsed object is
        ``None`` on failure.
    """

    stripped = raw_content.strip()
    warnings: list[str] = []
    if stripped.startswith("```"):
        warnings.append("raw output used markdown fence")
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, str(exc), warnings
    if not isinstance(parsed, dict):
        return None, "raw output was not a JSON object", warnings
    return parsed, "", warnings


async def run_case(case: LLMCase) -> dict[str, Any]:
    """Run one real-LLM prompt case and return a diagnostic record.

    Args:
        case: Case definition to execute.

    Returns:
        Diagnostic result containing raw and parsed model output.
    """

    module = importlib.import_module(case.module_name)
    prompt = getattr(module, case.prompt_name)
    llm = getattr(module, case.llm_name)
    if case.format_kwargs is not None:
        prompt = prompt.format(**case.format_kwargs)

    system_prompt = SystemMessage(content=prompt)
    if case.image_payload:
        human_content = [
            {
                "type": "image_url",
                "image_url": {"url": _tiny_png_data_uri()},
            }
        ]
    else:
        human_content = json.dumps(case.payload, ensure_ascii=False, default=str)
    human_message = HumanMessage(content=human_content)

    try:
        response = await llm.ainvoke([system_prompt, human_message])
    except Exception as exc:
        result = {
            "case_id": case.case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": case.module_name,
            "prompt_name": case.prompt_name,
            "llm_name": case.llm_name,
            "required_keys": list(case.required_keys),
            "input_payload": case.payload if not case.image_payload else {"image_payload": "tiny_png"},
            "raw_response": "",
            "parsed_response": None,
            "parse_error": "",
            "warnings": [],
            "missing_keys": list(case.required_keys),
            "passed": False,
            "judgment": "llm_call_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return result
    raw_content = str(response.content)
    parsed, parse_error, warnings = _strict_json_loads(raw_content)
    missing_keys = []
    if parsed is not None:
        missing_keys = [key for key in case.required_keys if key not in parsed]

    passed = parsed is not None and not missing_keys
    result = {
        "case_id": case.case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": case.module_name,
        "prompt_name": case.prompt_name,
        "llm_name": case.llm_name,
        "required_keys": list(case.required_keys),
        "input_payload": case.payload if not case.image_payload else {"image_payload": "tiny_png"},
        "raw_response": raw_content,
        "parsed_response": parsed,
        "parse_error": parse_error,
        "warnings": warnings,
        "missing_keys": missing_keys,
        "passed": passed,
        "judgment": (
            "accepted_structural_contract"
            if passed
            else "regression_or_contract_risk"
        ),
    }
    return result


def write_artifact(result: dict[str, Any]) -> Path:
    """Append one real-LLM result to the regression artifact.

    Args:
        result: Diagnostic result returned by ``run_case``.

    Returns:
        Artifact path.
    """

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("a", encoding="utf-8") as artifact_file:
        artifact_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return ARTIFACT_PATH


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list available case ids")
    parser.add_argument("--case", help="case id to run")
    return_value = parser.parse_args()
    return return_value


async def async_main() -> int:
    """Run the selected command.

    Returns:
        Process exit code.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    cases = _cases()
    if args.list:
        for case_id in cases:
            print(case_id)
        return 0
    if not args.case:
        print("Provide --case or --list")
        return 2
    if args.case not in cases:
        print(f"Unknown case: {args.case}")
        return 2

    result = await run_case(cases[args.case])
    artifact_path = write_artifact(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"artifact={artifact_path}")
    if result["passed"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
