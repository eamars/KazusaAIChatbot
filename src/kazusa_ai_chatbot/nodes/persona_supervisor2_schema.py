from typing import TypedDict
from kazusa_ai_chatbot.db import CharacterStateDoc


class GlobalPersonaState(TypedDict):
    # Character related
    character_state: CharacterStateDoc
    character_profile: dict

    # Inputs
    timestamp: str
    user_input: str
    user_id: str
    user_name: str
    bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # Bridge Variables (Outputs of stages)
    decontexualized_input: str  # from Stage 0
    research_facts: str       # From Stage 1
    internal_monologue: str   # From Stage 2
    character_intent: str      # From Stage 2 (Directives for Stage 3)
    final_speech: str         # From Stage 3
    
    # Persistent Persona Data
    mood_state: dict          # Carry-over emotion (e.g., {"anger": 2, "energy": 8})
    metadata: dict            # Search results, usage, etc.