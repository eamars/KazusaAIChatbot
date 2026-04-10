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
    user_affinity_score: int
    bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # Bridge Variables (Outputs of stages)
    decontexualized_input: str  # from Stage 0
    research_facts: str       # From Stage 1
    internal_monologue: str   # From Stage 2
    action_directives: list[str]      # From Stage 2 (Directives for Stage 3)
    final_action: str         # From Stage 3
