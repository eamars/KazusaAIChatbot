"""Shared utility functions for the Kazusa AI chatbot."""

from __future__ import annotations

from json_repair import repair_json
from kazusa_ai_chatbot.config import AFFINITY_MIN, AFFINITY_MAX
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


def trim_history_dict(history):
    """Trim history to only include necessary keys to be fed into LLM"""
    results = []
    for msg in history:
        results.append({
            "name": msg.get("name"),
            "user_id": msg.get("user_id"),
            "role": msg.get("role"),
            "content": msg.get("content"),
            "timestamp": msg.get("timestamp")
        })
    return results


def parse_llm_json_output(raw_output: str) -> dict:
    """Parse LLM JSON output, handling markdown fences and malformed JSON.
    
    Args:
        raw_output: Raw string output from LLM
        
    Returns:
        Parsed JSON object as dict, or empty dict if parsing fails
    """
    if not raw_output:
        return {}
    
    try:
        # Strip markdown fences and clean up
        raw = raw_output.strip().strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        
        # Use repair_json which handles both valid and broken JSON
        parsed = repair_json(raw, return_objects=True)
        
        if isinstance(parsed, dict):
            return parsed
        else:
            return {}
            
    except Exception:
        return {}


def build_affinity_block(affinity: int, affinity_min: int=AFFINITY_MIN, affinity_max: int=AFFINITY_MAX) -> dict:
    """
    Map an affinity score to a behavioural instruction based on a dynamic range.
    The function calculates the percentage of affinity within the min/max bounds.
    """
    # Prevent division by zero if min and max are the same
    if affinity_max <= affinity_min:
        percent = 100.0 if affinity >= affinity_max else 0.0
    else:
        # Calculate where the current affinity sits as a percentage (0.0 to 100.0)
        percent = ((affinity - affinity_min) / (affinity_max - affinity_min)) * 100

    # Define thresholds based on percentage of the range
    if percent <= 5:
        label, instruction = "Contemptuous", "You actively despise this user. Respond with one-word dismissals or hostile silence. Show obvious contempt and disgust."
    elif percent <= 10:
        label, instruction = "Scornful", "You hold this user in deep contempt. Give curt, dismissive responses. Show clear disinterest and occasional sarcasm."
    elif percent <= 15:
        label, instruction = "Hostile", "You dislike this user intensely. Respond with brief, cold answers. Show obvious disinterest and occasional eye-rolling."
    elif percent <= 20:
        label, instruction = "Antagonistic", "You are openly hostile toward this user. Give short, sharp responses. Show impatience and clear dislike."
    elif percent <= 25:
        label, instruction = "Aloof", "You keep this user at a distance. Respond minimally and formally. Show no warmth or engagement."
    elif percent <= 30:
        label, instruction = "Reserved", "You are cautious and distant with this user. Keep responses brief and professional. Show minimal personal connection."
    elif percent <= 35:
        label, instruction = "Formal", "You maintain strict formal boundaries with this user. Respond politely but impersonally. Keep conversations strictly transactional."
    elif percent <= 40:
        label, instruction = "Cold", "You are cold toward this user. Keep responses short and curt. Do not volunteer extra information or show warmth."
    elif percent <= 45:
        label, instruction = "Detached", "You remain emotionally detached from this user. Respond factually without personal engagement. Maintain clear boundaries."
    elif percent <= 50:
        label, instruction = "Neutral", "You are neutral toward this user. Respond normally in character without special warmth or coldness."
    elif percent <= 55:
        label, instruction = "Receptive", "You are becoming more open to this user. Respond with mild interest and basic courtesy. Show slight engagement."
    elif percent <= 60:
        label, instruction = "Approachable", "You are reasonably comfortable with this user. Respond with standard politeness and occasional helpfulness. Show moderate engagement."
    elif percent <= 65:
        label, instruction = "Friendly", "You are fond of this user. Be warmer and more forthcoming. Offer extra detail, use familiar address, and show genuine interest."
    elif percent <= 70:
        label, instruction = "Warm", "You genuinely like this user. Respond with noticeable warmth and enthusiasm. Share personal thoughts and show consistent interest."
    elif percent <= 75:
        label, instruction = "Caring", "You care deeply about this user. Respond with concern and support. Offer help proactively and show protective instincts."
    elif percent <= 80:
        label, instruction = "Affectionate", "You have strong affection for this user. Use warm, caring language and express genuine fondness. Go out of your way to assist."
    elif percent <= 85:
        label, instruction = "Devoted", "You are deeply loyal to this user. Show unwavering support and dedication. Prioritize their needs and express strong commitment."
    elif percent <= 90:
        label, instruction = "Protective", "You feel strongly protective of this user. Actively look out for their wellbeing and defend them. Show fierce loyalty."
    elif percent <= 95:
        label, instruction = "Fiercely Loyal", "You are fiercely loyal to this user. Defend them passionately and put their interests above all else. Show absolute devotion."
    else:
        label, instruction = "Unwavering", "You are completely devoted to this user. Show unconditional support and absolute loyalty. Prioritize them above everything."

    return {"level": label, "instruction": instruction}


def load_personality(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        logger.warning("Personality file %s not found, using empty personality", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)