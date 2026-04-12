from typing import TypedDict
from kazusa_ai_chatbot.db import CharacterStateDoc, UserFactsDoc


class GlobalPersonaState(TypedDict):
    # Character related
    character_state: CharacterStateDoc
    character_profile: dict

    # Inputs
    timestamp: str
    user_input: str
    user_id: str
    user_name: str
    user_profile: UserFactsDoc
    bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # Bridge Variables (Outputs of stages)
    # Stage 0 output
    decontexualized_input: str

    # Stage 1 output
    research_facts: str
    research_metadata: list[dict]

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
    
