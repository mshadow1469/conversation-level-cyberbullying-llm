# src/finetuning/lorma/llama_lorma_plus_a0_bkaiming.py
"""
LoRMA+ fine-tuning, Llama-3.2-3B-Instruct on cyberbullying detection.
Initialisation: A = 0, B = Kaiming-uniform  (reference-code convention)
All hyperparameters matched to llama_lora.py for a fair LoRA-vs-LoRMA+
comparison. 

    LoRA:   W' = W0 + (alpha/r) * B * A            [additive update]
    LoRMA+: W' = W0 + (alpha/r) * B * (A * W0)     [multiplicative update]

Reference:
    Bihany, Patel & Modi (2025). LoRMA: Low-Rank Multiplicative Adaptation
    for LLMs. ACL Findings 2025
    https://github.com/Exploration-Lab/LoRMA
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

from src.framework.data import extract_conversation_text
from src.framework.prompts import BASE_PROMPT
from src.framework.evaluate import evaluate_predictions, print_results, save_results

LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_MAX_NEW_TOKENS = 6
CACHE_DIR = "/scratch/mb02997/hf"
DATA_DIR = Path("data_processed")
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = Path("outputs/llama_cb2_lorma_a0_bkaiming")
ADAPTER_DIR = OUTPUT_DIR / "final_adapter"
LORMA_STATE_PATH = ADAPTER_DIR / "lorma_plus_state.pt"

LABEL_COL = "label"

NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LENGTH = 6144

LORMA_R = 16
LORMA_ALPHA = 32
LORMA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

# Run control
RUN_TRAIN = True
RUN_EVAL = True

# PROMPT CONSTRUCTION
# zero-shot only

def build_zero_shot_messages(conversation_text: str):
    """Build the chat-template messages used at inference time."""
    return [
        {"role": "system", "content": BASE_PROMPT},
        {
            "role": "user",
            "content": f"Conversation:\n{conversation_text}\n\nAnswer with only 0 or 1."
        },
    ]


def build_train_example(conversation_text: str, label: int, tokenizer):
    """
    Build one supervised fine-tuning example.

    Returns a dict with 'prompt' and 'completion' fields. SFTTrainer is
    configured with completion_only_loss=True (see train()), so the loss
    is computed only on the 'completion' (the single label character +
    EOS), while the 'prompt' is treated as fixed context.
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
    """Convert a DataFrame of conversations into a HuggingFace Dataset."""
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

# LoRMA+ LAYER  (reference-code init)
# Corresponds to: model_lorma_plus.py in official LoRMA repository,
# LoRMA+ block inside the GPT-2 Attention class.
# Reference snippet (Exploration-Lab/LoRMA/NLG/Language-generation/src/
# model_lorma_plus.py, inside Attention.forward):
class LoRMAPlusLinear(nn.Module):
    """
    Wraps existing nn.Linear and adds a LoRMA+ multiplicative
    low-rank adaptation path computed in weight space.
    """

    def __init__(
        self,
        base_linear: nn.Module,
        r: int,
        alpha: float,
    ):
        super().__init__()

        if not hasattr(base_linear, "weight"):
            raise TypeError(
                "LoRMAPlusLinear expects a module with a .weight attribute"
            )

        # Keeps the frozen base layer
        # via mark_only_lorma_as_trainable below.
        self.base_linear = base_linear
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0

        self.out_features = base_linear.out_features
        self.in_features = base_linear.in_features

        # Reference-code initialisation:
        #   A = 0, B = Kaiming-uniform
        self.lora_A = nn.Parameter(torch.zeros(r, self.out_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W has shape (out_features, in_features)
        W = self.base_linear.weight
        bias = self.base_linear.bias
        # Cast adapter parameters to the base weight's dtype/device.
        # parameters in their stored fp32 form.
        A = self.lora_A.to(dtype=W.dtype, device=W.device)
        B = self.lora_B.to(dtype=W.dtype, device=W.device)

        # adapted_W = W + scaling * B @ (A @ W)
        #   A @ W : (r, in_features)
        #   B @ . : (out_features, in_features)
        # Final shape matches W exactly.
        adapted_W = W + self.scaling * (B @ (A @ W))

        return F.linear(x, adapted_W, bias)


# MODEL PATCHING
# After loading the base model, replaces every
# Linear layer whose  path ends with one of the target projection
# names (q_proj, k_proj, v_proj, o_proj) with a LoRMAPlusLinear wrapper
# around the original layer.

def _get_parent_module(model: nn.Module, full_name: str):
    """Traverse the model.named_modules() dot-path and return (parent_module, child_name)."""
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lorma_plus(
    model: nn.Module,
    target_names: Iterable[str] = LORMA_TARGET_MODULES,
    r: int = LORMA_R,
    alpha: float = LORMA_ALPHA,
) -> nn.Module:
    """
    Finds the selected projection layers and replaces them with
    LoRMAPlusLinear wrappers. For Llama 3.2-3B this covers 28 layers
    and 4 projections per layer, so 112 modules are adapted.
    """
    to_replace = []

    # First pass collects (name, module) pairs
    for name, module in model.named_modules():
        if name.endswith(tuple(target_names)) and hasattr(module, "weight"):
            to_replace.append((name, module))

    # Second pass replaces each matched projection with the LoRMA+ wrapper.
    for name, module in to_replace:
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            LoRMAPlusLinear(base_linear=module, r=r, alpha=alpha),
        )

    print(f"LoRMA+: wrapped {len(to_replace)} layers (r={r}, alpha={alpha})")
    return model


def mark_only_lorma_as_trainable(model: nn.Module) -> None:
    """
    Freezes every parameter except the LoRMA+ adapter matrices.
    Corresponds to mark_only_lora_as_trainable() in
    gpt2_ft_lorma_plus.py (Exploration-Lab/LoRMA repo). 
    Then Identifies LoRMA+ parameters by substring 'lora_A' or 'lora_B'
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def lorma_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Extracts only LoRMA+ adapter parameters from the full state dict
    corresponds to: mult_lora_state_dict() in gpt2_ft_lorma_plus.py
    (Exploration-Lab/LoRMA repo).
    """
    state = model.state_dict()
    return {k: v.cpu() for k, v in state.items() if "lora_A" in k or "lora_B" in k}


def print_trainable_params(model: nn.Module) -> None:
    """Prints trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


# MODEL LOADING

def get_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_train_model():
    """Loads tokenizer and base model, then attaches LoRMA+ wrappers."""
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

    model = apply_lorma_plus(
        model,
        target_names=LORMA_TARGET_MODULES,
        r=LORMA_R,
        alpha=LORMA_ALPHA,
    )

    mark_only_lorma_as_trainable(model)
    print_trainable_params(model)

    return tokenizer, model


def load_eval_model():
    """
    Reloads frozen base model, attaches LoRMAPlusLinear wrappers,
    then loads saved adapter parameters on top.
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

    base_model = apply_lorma_plus(
        base_model,
        target_names=LORMA_TARGET_MODULES,
        r=LORMA_R,
        alpha=LORMA_ALPHA,
    )

    state = torch.load(LORMA_STATE_PATH, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    print(f"Loaded LoRMA+ state. Missing keys: {len(missing)}, "
          f"unexpected keys: {len(unexpected)}")

    base_model.eval()
    return tokenizer, base_model


# TRAINING

def train():
    """Run the full SFT training loop using TRL SFTTrainer."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    print(f"Train size: {len(train_df)}")
    print(train_df[LABEL_COL].value_counts(dropna=False).sort_index())
    print()
    print(f"Val size: {len(val_df)}")
    print(val_df[LABEL_COL].value_counts(dropna=False).sort_index())
    print()

    tokenizer, model = load_train_model()

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
        save_strategy="no",        # LoRMA+ adapters saved manually below
        logging_steps=10,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True, 
        packing=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save only the LoRMA+ adapter parameters (not the frozen base model)
    torch.save(lorma_state_dict(model), LORMA_STATE_PATH)
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Saved LoRMA+ state to {LORMA_STATE_PATH}")


# INFERENCE

def predict(model, tokenizer, messages):
    """Greedy 6-token generation; parse the first character as 0 or 1."""
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
    """Load the trained adapter and evaluate on the held-out test set."""
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
        results, "llama_lorma_a0_bkaiming", "zero_shot",
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