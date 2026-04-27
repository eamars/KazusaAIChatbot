from typing import TypedDict
from kazusa_ai_chatbot.state import MultiMediaDoc, DebugModes, ReplyContext
from kazusa_ai_chatbot.db import CharacterProfileDoc, UserProfileDoc


class GlobalPersonaState(TypedDict):
    # Character related
    character_profile: CharacterProfileDoc

    # Inputs
    timestamp: str
    user_input: str
    user_multimedia_input: list[MultiMediaDoc]
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    platform_user_id: str
    global_user_id: str
    user_name: str
    user_profile: UserProfileDoc
    platform_bot_id: str
    chat_history_wide: list[dict]
    chat_history_recent: list[dict]
    reply_context: ReplyContext
    indirect_speech_context: str
    channel_topic: str

    # Debug
    debug_modes: DebugModes

    # Bridge Variables (Outputs of stages)
    # Stage 0 output
    decontexualized_input: str

    # Stage 1 output
    rag_result: dict

    # Stage 2 output
    internal_monologue: str
    action_directives: dict

    # Stage 2 output for stage 4 consolidation
    interaction_subtext: str
    emotional_appraisal: str
    character_intent: str
    logical_stance: str

    # Stage 3 output
    final_dialog: [str]  # -> Will be used for dialog end point (e.g,. Discord)
    # Other outputs from here

    # Stage 4 output
    # global state updater
    mood: str
    global_vibe: str
    reflection_summary: str

    # Relationship recorder
    diary_entry: [str]
    affinity_delta: int
    last_relationship_insight: str

    # Facts harvester
    new_facts: [str]
    future_promises: [str]
    

class CognitionState(TypedDict):
    character_profile: CharacterProfileDoc

    timestamp: str
    user_input: str
    global_user_id: str
    user_name: str
    user_profile: UserProfileDoc
    platform_bot_id: str
    chat_history_recent: list[dict]
    reply_context: ReplyContext
    indirect_speech_context: str
    channel_topic: str

    decontexualized_input: str
    rag_result: dict

    emotional_appraisal: str
    interaction_subtext: str

    internal_monologue: str
    character_intent: str
    logical_stance: str
    judgment_note: str

    boundary_core_assessment: dict

    social_distance: str
    emotional_intensity: str
    vibe_check: str
    relational_dynamic: str
    expression_willingness: str

    rhetorical_strategy: str
    linguistic_style: str
    accepted_user_preferences: list[str]
    forbidden_phrases: list[str]

    content_anchors: list[str]

    facial_expression: list[str]
    body_language: list[str]
    gaze_direction: list[str]
    visual_vibe: list[str]

    action_directives: dict

    should_stop: bool
    reasoning: str
    retry: int
