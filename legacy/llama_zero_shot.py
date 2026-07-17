# legacy/llama_zero_shot.py
"""
Standalone zero-shot Llama script
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_MAX_NEW_TOKENS = 6

DATA_DIR = Path("data_processed")

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

LABEL_COL = "label"

MAX_MESSAGES = 50
MAX_CONV_CHARS = 30000


# Prompt
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


def build_zero_shot_messages(conversation_text: str):
    return [
        {"role": "system", "content": BASE_PROMPT},
        {
            "role": "user",
            "content": f"Conversation:\n{conversation_text}\n\nAnswer with only 0 or 1."
        },
    ]


# Data processing
def parse_conversation(conv):
    if pd.isna(conv):
        return []

    data = json.loads(conv)
    return [m["message"] for m in data if "message" in m]


def format_conversation(messages):
    messages = messages[:MAX_MESSAGES]
    text = "\n".join(messages)

    if len(text) > MAX_CONV_CHARS:
        text = text[:MAX_CONV_CHARS].rstrip() + "..."

    return text


def extract_prompt_text_from_row(row):
    messages = parse_conversation(row["conversation"])
    return format_conversation(messages)


# Model
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def predict(model, tokenizer, conversation_text):

    messages = build_zero_shot_messages(conversation_text)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    prediction_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if prediction_text.startswith("1"):
        return 1
    if prediction_text.startswith("0"):
        return 0

    return 0


# Main
def main():

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    test_df = pd.read_csv(TEST_PATH)
    tokenizer, model = load_model()

    preds = []
    labels = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

        conversation_text = extract_prompt_text_from_row(row)

        pred = predict(model, tokenizer, conversation_text)

        preds.append(pred)
        labels.append(row[LABEL_COL])

    print("\nResults\n")

    print(classification_report(labels, preds))

    print("Macro F1:", f1_score(labels, preds, average="macro"))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()