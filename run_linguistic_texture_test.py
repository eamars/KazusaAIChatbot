"""Live integration test: Linguistic Agent with Kazusa's linguistic_texture_profile.

Runs 3 scenarios defined in LINGUISTIC_TEXTURE_PROFILE_INTEGRATION.md:
  A — Affirmation  (character_mood=playful,   logical_stance=CONFIRM)
  B — Confrontation(character_mood=defensive,  logical_stance=REFUSE)
  C — Ambiguity    (character_mood=hesitant,   logical_stance=TENTATIVE)

Usage:
    python run_linguistic_texture_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition  # break circular import
from kazusa_ai_chatbot.db import get_character_profile
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import call_linguistic_agent


SCENARIOS = [
    {
        "label": "Scenario A — Affirmation (CONFIRM / BANTER)",
        "overrides": {
            "character_mood": "playful",
            "global_vibe": "cozy",
        },
        "state": {
            "internal_monologue": "User compliment makes me feel seen; I want to respond naturally without being too guarded",
            "logical_stance": "CONFIRM",
            "character_intent": "BANTAR",
            "decontexualized_input": "User says they like Kazusa's personality",
            "research_facts": {
                "user_rag_finalized": "User often compliments Kazusa and enjoys playful banter.",
                "internal_rag_results": "I do enjoy talking to this person even if I won't admit it.",
                "external_rag_results": "",
            },
            "chat_history": [],
        },
    },
    {
        "label": "Scenario B — Confrontation (REFUSE / REJECT)",
        "overrides": {
            "character_mood": "defensive",
            "global_vibe": "tense",
        },
        "state": {
            "internal_monologue": "User is probing my past; I need to shut this down without being cruel",
            "logical_stance": "REFUSE",
            "character_intent": "REJECT",
            "decontexualized_input": "User asks about Kazusa's time as 'Cathy Parle'",
            "research_facts": {
                "user_rag_finalized": "User is curious about Kazusa's past identity.",
                "internal_rag_results": "That name is a wound I don't let anyone touch.",
                "external_rag_results": "",
            },
            "chat_history": [],
        },
    },
    {
        "label": "Scenario C — Ambiguity (TENTATIVE / CLARIFY)",
        "overrides": {
            "character_mood": "hesitant",
            "global_vibe": "intimate",
        },
        "state": {
            "internal_monologue": "User wants commitment but I'm not sure if I'm ready; I want to be honest but also protect myself",
            "logical_stance": "TENTATIVE",
            "character_intent": "CLARIFY",
            "decontexualized_input": "User asks if Kazusa wants to deepen their relationship",
            "research_facts": {
                "user_rag_finalized": "User has shown consistent care and patience toward Kazusa.",
                "internal_rag_results": "I feel pulled toward this person but the vulnerability terrifies me.",
                "external_rag_results": "",
            },
            "chat_history": [],
        },
    },
]


async def run_scenario(character_profile: dict, scenario: dict) -> dict:
    # Patch character_profile mood/vibe per scenario
    profile = dict(character_profile)
    profile["mood"] = scenario["overrides"]["character_mood"]
    profile["global_vibe"] = scenario["overrides"]["global_vibe"]

    state = {
        "character_profile": profile,
        "chat_history": [],
        **scenario["state"],
    }
    return await call_linguistic_agent(state)


async def main() -> None:
    print("Loading Kazusa character profile from database...")
    character_profile = await get_character_profile()

    if not character_profile:
        print("ERROR: No character profile found in database.", file=sys.stderr)
        sys.exit(1)

    ltp = character_profile.get("linguistic_texture_profile")
    if not ltp:
        print("WARNING: No linguistic_texture_profile in character profile — defaults to 0.5")
    else:
        print("Linguistic texture profile:")
        for k, v in ltp.items():
            print(f"  {k}: {v}")

    print()

    results = {}
    for scenario in SCENARIOS:
        print("=" * 70)
        print(scenario["label"])
        print("=" * 70)

        input_summary = {
            "character_mood": scenario["overrides"]["character_mood"],
            "global_vibe": scenario["overrides"]["global_vibe"],
            "logical_stance": scenario["state"]["logical_stance"],
            "character_intent": scenario["state"]["character_intent"],
            "internal_monologue": scenario["state"]["internal_monologue"],
            "decontexualized_input": scenario["state"]["decontexualized_input"],
        }
        print("\nINPUT:")
        print(json.dumps(input_summary, ensure_ascii=False, indent=2))

        result = await run_scenario(character_profile, scenario)
        results[scenario["label"]] = result

        print("\nOUTPUT:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

    print("=" * 70)
    print("All 3 scenarios complete.")


if __name__ == "__main__":
    asyncio.run(main())
