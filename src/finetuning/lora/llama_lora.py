# src/finetuning/lora/llama_lora.py
"""
LoRA fine-tuning for Llama-3.2-3B-Instruct on CB2 cyberbullying detection.
Implemented as LoRA baseline against which the two
LoRMA+ initialisation variants (llama_lorma_plus_akaiming_b0.py and
llama_lorma_plus_a0_bkaiming.py) are compared.

All hyperparameters match the LoRMA+ scripts so the three runs differ
only in the adaptation method and (for the LoRMA+ pair) the
initialisation scheme.

    LoRA:   W' = W0 + (alpha/r) * B * A            [additive update]
    LoRMA+: W' = W0 + (alpha/r) * B * (A * W0)     [multiplicative update]

Reference:
    Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
    ICLR 2022 https://arxiv.org/abs/2106.09685
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    LoraConfig,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig

from src.framework.data import extract_conversation_text
from src.framework.prompts import BASE_PROMPT
from src.framework.evaluate import evaluate_predictions, print_results, save_results

# CONFIG 

LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_MAX_NEW_TOKENS = 6
CACHE_DIR = "/scratch/mb02997/hf"
DATA_DIR = Path("data_processed")
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = Path("outputs/llama_cb2_lora")
ADAPTER_DIR = OUTPUT_DIR / "final_adapter"

LABEL_COL = "label"

NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8

MAX_SEQ_LENGTH = 6144

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

RUN_TRAIN = True
RUN_EVAL = True


# PROMPT CONSTRUCTION
# zero-shot only

def build_zero_shot_messages(conversation_text: str):
    """Builds chat-template messages used at inference time"""
    return [
        {"role": "system", "content": BASE_PROMPT},
        {
            "role": "user",
            "content": f"Conversation:\n{conversation_text}\n\nAnswer with only 0 or 1."
        },
    ]


def build_train_example(conversation_text: str, label: int, tokenizer):
    """
    Builds one supervised fine-tuning example
    """
    messages = [
        {"role": "system", "content": BASE_PROMPT},
        {
            "role": "user",
            "content": f"Conversation:\n{conversation_text}\n\nAnswer with only 0 or 1."
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt_text,
        "completion": str(int(label)) + tokenizer.eos_token,
    }


def make_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Converts DataFrame of conversations into a HuggingFace Dataset"""
    rows = []
    for _, row in df.iterrows():
        conversation_text = extract_conversation_text(row)
        rows.append(
            build_train_example(
                conversation_text,
                int(row[LABEL_COL]),
                tokenizer,
            )
        )
    return Dataset.from_list(rows)

# MODEL LOADING

def get_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_train_model():
    """
    Loads tokenizer and base model, then returns them together with
    LoraConfig
    """
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_ID,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=get_dtype(),
        cache_dir=CACHE_DIR,
    )

    # Target modules match LoRMA+ runs 
    # (q_proj, k_proj, v_proj,o_proj)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    return tokenizer, model, peft_config


def load_eval_model():
    """
    Reloads frozen base model and attaches saved LoRA adapter
    """
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_ID,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        device_map="auto",
        torch_dtype=get_dtype(),
        cache_dir=CACHE_DIR,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return tokenizer, model

# TRAINING 

def train():
    """Runs full SFT training loop using TRL SFTTrainer + PEFT LoRA."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    print(f"Train size: {len(train_df)}")
    print(train_df[LABEL_COL].value_counts(dropna=False).sort_index())
    print()
    print(f"Val size: {len(val_df)}")
    print(val_df[LABEL_COL].value_counts(dropna=False).sort_index())
    print()

    tokenizer, model, peft_config = load_train_model()

    train_dataset = make_dataset(train_df, tokenizer)
    val_dataset = make_dataset(val_df, tokenizer)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        packing=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    # peft_config passed here, installs the LoRA adapters
    # on the model and freezes the base weights automatically.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

# INFERENCE

def predict(model, tokenizer, messages):
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
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


def evaluate():
    """Loads trained LoRA adapter and evaluates on held-out test set."""
    test_df = pd.read_csv(TEST_PATH)

    print(f"Test size: {len(test_df)}")
    print(test_df[LABEL_COL].value_counts(dropna=False).sort_index())
    print()

    tokenizer, model = load_eval_model()

    preds = []
    labels = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        conversation_text = extract_conversation_text(row)
        messages = build_zero_shot_messages(conversation_text)
        pred = predict(model, tokenizer, messages)
        preds.append(pred)
        labels.append(int(row[LABEL_COL]))

    results = evaluate_predictions(labels, preds)
    print_results(results)
    save_results(
        results, "llama_lora", "zero_shot",
        Path("outputs/results"), preds=preds, labels=labels,
    )

# MAIN

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    if RUN_TRAIN:
        train()
        torch.cuda.empty_cache()

    if RUN_EVAL:
        evaluate()


if __name__ == "__main__":
    main()