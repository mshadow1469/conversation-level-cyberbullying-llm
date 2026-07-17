# src/framework/data.py
"""
Data loading and conversation parsing.
Used by every experiment, no duplication.
"""
from __future__ import annotations

import json

import pandas as pd

from src.framework.config import (
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    LABEL_COL, MAX_MESSAGES, MAX_CONV_CHARS,
)

# Required columns, validated on load
REQUIRED_COLUMNS = [
    "user1_id", "user2_id", "total_messages",
    "aggressive_count", "intent_to_harm", "peerness",
    "label", "conversation",
]


# Loads splits
def load_splits():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    return train_df, val_df, test_df


# Conversation parsing
def parse_conversation(conv) -> list[str]:
    if pd.isna(conv):
        return []
    data = json.loads(conv)
    return [item["message"] for item in data if "message" in item]


# Truncates and joins messages into a single text block
def format_conversation(messages: list[str], max_chars: int = MAX_CONV_CHARS) -> str:
    messages = messages[:MAX_MESSAGES]
    text = "\n".join(messages)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


# Extracts and formats conversation text from one DataFrame row
def extract_conversation_text(row, max_chars: int = MAX_CONV_CHARS) -> str:
    messages = parse_conversation(row["conversation"])
    return format_conversation(messages, max_chars=max_chars)


# Extracts metadata fields from one DataFrame row
def extract_metadata(row) -> dict:
    def safe_float(val):
        if pd.isna(val):
            return None
        return float(val)

    def safe_int(val):
        if pd.isna(val):
            return None
        return int(val)

    return {
        "peerness": safe_float(row.get("peerness")),
        "intent_to_harm": safe_float(row.get("intent_to_harm")),
        "aggressive_count": safe_int(row.get("aggressive_count")),
    }
