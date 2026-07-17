# src/framework/config.py
"""
Central configuration for inference for all models.
Every model-specific detail lives here.
"""
from pathlib import Path

# Paths
DATA_DIR = Path("data_processed")
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
OUTPUT_DIR = Path("outputs")

# Dataset schema
LABEL_COL = "label"
CONV_COL = "conversation"

# Conversation truncation
MAX_MESSAGES = 50
MAX_CONV_CHARS = 30000

# Model registry
# backend: "hf" (HuggingFace transformers) or "gguf" (llama.cpp)

MODELS = {
    "llama": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "backend": "hf",
        "max_new_tokens": 6,
        "use_4bit": False,
        "torch_dtype": "float16",
    },
    "gemma": {
        "model_id": "/scratch/mb02997/models/gemma-3-27b-it.IQ4_XS.gguf",
        "backend": "gguf",
        "max_new_tokens": 6,
        "n_ctx": 6944,
        "n_gpu_layers": -1,
        "max_conv_chars": 12000,
        "max_example_chars": 1500,
    },
    "qwen": {
        "model_id": "Qwen/Qwen3.5-9B",
        "backend": "hf",
        "max_new_tokens": 6,
        "use_4bit": True,
        "torch_dtype": "float16",
        "enable_thinking": False,
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "backend": "hf",
        "max_new_tokens": 6,
        "use_4bit": False,
        "torch_dtype": "float16",
    },
}

# Valid strategy names

VALID_STRATEGIES = [
    "zero_shot",
    "few_shot",
    "zero_shot_peer",
    "few_shot_peer",
    "zero_shot_intent",
    "zero_shot_aggressive",
    "zero_shot_meta",
    "few_shot_meta",
]
