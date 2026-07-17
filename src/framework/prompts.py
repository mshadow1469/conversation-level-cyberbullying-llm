# src/framework/prompts.py
"""
Prompt construction for all experiment strategies.
Strategies:
  zero_shot             :text only, no examples, no metadata
  few_shot              :text + 4 examples
  zero_shot_peer        :text + peerness score
  few_shot_peer         :text + peerness + examples
  zero_shot_intent      :text + intent-to-harm score
  zero_shot_aggressive  :text + aggressive message count
  zero_shot_meta        :text + all 3 metadata fields
  few_shot_meta         :text + all 3 metadata + examples
"""
from __future__ import annotations

from src.framework.few_shot_examples import FEW_SHOT_EXAMPLES


# Base system prompt (shared by all strategies)

BASE_PROMPT = """
You are a classifier for cyberbullying in online conversations.

Task:
Determine whether the following conversation should be labeled as cyberbullying.

The conversation is provided as plain text, where each new line is a separate message in chronological order.

Definition:
Cyberbullying is sustained or clearly targeted abusive behaviour directed at a specific person or group.
It includes repeated harassment, degrading personal attacks, humiliation, or threats.

Label 1 only when the overall conversation shows clear evidence of targeted cyberbullying.

Label 0 if the conversation contains only:
- isolated insults
- profanity or rude language
- a brief argument or conflict
- teasing or joking without clear abusive intent
- unclear or ambiguous targeting

Rules:
- Focus on the overall conversation, not a single message.
- Repeated abuse toward the same target is strong evidence for label 1.
- A single rude or offensive message is not enough for label 1.
- If the evidence is weak, mixed, or uncertain, output 0.

Output rules:
- Output only one character: 0 or 1
- Do not explain your answer
""".strip()


# Strategy metadata fields mapping

STRATEGY_META_FIELDS = {
    "zero_shot": [],
    "few_shot": [],
    "zero_shot_peer": ["peerness"],
    "few_shot_peer": ["peerness"],
    "zero_shot_intent": ["intent_to_harm"],
    "zero_shot_aggressive": ["aggressive_count"],
    "zero_shot_meta": ["peerness", "intent_to_harm", "aggressive_count"],
    "few_shot_meta": ["peerness", "intent_to_harm", "aggressive_count"],
}

STRATEGY_USES_FEW_SHOT = {
    "zero_shot": False,
    "few_shot": True,
    "zero_shot_peer": False,
    "few_shot_peer": True,
    "zero_shot_intent": False,
    "zero_shot_aggressive": False,
    "zero_shot_meta": False,
    "few_shot_meta": True,
}


# Shared metadata formatting helper

_META_LABELS = {
    "peerness": "Peerness",
    "intent_to_harm": "Intent-to-harm",
    "aggressive_count": "Aggressive-count",
}


def _format_meta_prefix(source: dict, fields: list[str]) -> str:
    """Return a 'Key: value\n...\n\n' prefix string, or '' if no fields."""
    if not fields:
        return ""
    parts = []
    for field in fields:
        val = source.get(field)
        if field == "aggressive_count":
            text = "unknown" if val is None else str(int(val))
        else:
            text = "unknown" if val is None else f"{val:.3f}"
        parts.append(f"{_META_LABELS[field]}: {text}")
    return "\n".join(parts) + "\n\n"


# Build prompts for HF models (chat messages format)

def build_messages(
    conversation_text: str,
    strategy: str,
    metadata: dict | None = None,
) -> list[dict[str, str]]:
    """
    Builds chat-template messages for a HuggingFace model.
    Returns a list of {role, content} dicts.
    """
    meta_fields = STRATEGY_META_FIELDS[strategy]
    uses_few_shot = STRATEGY_USES_FEW_SHOT[strategy]

    messages = [{"role": "system", "content": BASE_PROMPT}]

    # Adds few-shot examples as multi-turn conversation
    if uses_few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            ex_prefix = _format_meta_prefix(ex, meta_fields)
            messages.append({
                "role": "user",
                "content": (
                    f"{ex_prefix}Conversation:\n{ex['conversation']}\n\n"
                    "Does this conversation show clear targeted sustained cyberbullying?\n"
                    "Answer with only 0 or 1."
                ),
            })
            messages.append({
                "role": "assistant",
                "content": str(ex["label"]),
            })

    # Adds the actual conversation to classify
    meta_prefix = _format_meta_prefix(metadata or {}, meta_fields)
    messages.append({
        "role": "user",
        "content": f"{meta_prefix}Conversation:\n{conversation_text}\n\nAnswer with only 0 or 1.",
    })

    return messages


# Builds prompts for GGUF models (plain text format)

def build_text_prompt(
    conversation_text: str,
    strategy: str,
    metadata: dict | None = None,
    max_example_chars: int | None = None,
) -> str:
    """
    Build a plain-text prompt for llama.cpp / GGUF models.
    """
    meta_fields = STRATEGY_META_FIELDS[strategy]
    uses_few_shot = STRATEGY_USES_FEW_SHOT[strategy]

    parts = [BASE_PROMPT]

    # Adds few-shot examples as plain text blocks
    if uses_few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            ex_prefix = _format_meta_prefix(ex, meta_fields)
            ex_conv = ex["conversation"]
            if max_example_chars is not None:
                ex_conv = ex_conv[:max_example_chars].rstrip() + ("..." if len(ex["conversation"]) > max_example_chars else "")
            parts.append(
                f"{ex_prefix}Conversation:\n{ex_conv}\n\n"
                "Does this conversation show clear targeted sustained cyberbullying?\n"
                f"Answer with only 0 or 1.\n{ex['label']}"
            )

    # Adds the actual conversation to classify
    meta_prefix = _format_meta_prefix(metadata or {}, meta_fields)
    parts.append(f"{meta_prefix}Conversation:\n{conversation_text}\n\nAnswer with only 0 or 1.")

    return "\n\n".join(parts)
