# src/framework/models.py
"""
Model loading and inference for HuggingFace and GGUF backends.
Provides unified interface so experiment runner does not
need to know which backend is being used.
"""
from __future__ import annotations

import re

import torch
from src.framework.config import MODELS

# Loads HuggingFace model and tokenizer
def load_hf_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    cfg = MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, cfg.get("torch_dtype", "float16"))

    if cfg.get("use_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            device_map="auto",
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            device_map="auto",
            dtype=dtype,
        )

    return model, tokenizer


# Runs inference on a HuggingFace model using chat template
def predict_hf(model, tokenizer, messages: list[dict], max_new_tokens: int = 6, enable_thinking: bool = True) -> int:
    template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    if not enable_thinking:
        template_kwargs["enable_thinking"] = False

    inputs = tokenizer.apply_chat_template(messages, **template_kwargs).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    prediction_text = tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    ).strip()

    return _parse_prediction(prediction_text)


# GGUF backend (llama.cpp)

# Loads a GGUF model via llama-cpp-python
def load_gguf_model(model_name: str):
    from llama_cpp import Llama

    cfg = MODELS[model_name]

    model = Llama(
        model_path=cfg["model_id"],
        n_ctx=cfg.get("n_ctx", 6144),
        n_gpu_layers=cfg.get("n_gpu_layers", -1),
        verbose=False,
    )

    return model, None  # no separate tokenizer for GGUF


# Runs inference on a GGUF model using plain text prompt
def predict_gguf(model, prompt: str, max_new_tokens: int = 6) -> int:
    output = model(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        echo=False,
    )

    prediction_text = output["choices"][0]["text"].strip()
    return _parse_prediction(prediction_text)


# Output parsing (shared)

def _parse_prediction(text: str) -> int:
    # Direct prefix match
    if text.startswith("1"):
        return 1
    if text.startswith("0"):
        return 0

    # Line-by-line check for models that output extra text before the label
    for line in text.splitlines():
        line = line.strip()
        if line in ("0", "0.", "1", "1."):
            return int(line[0])

    # Regex fallback
    match = re.search(r'(?m)^\s*([01])(?:\.)?\s*$', text)
    if match:
        return int(match.group(1))

    # Default to majority class
    return 0

# Loads model by name returns (model, tokenizer), tokenizer is None for GGUF
def load_model(model_name: str):
    cfg = MODELS[model_name]

    if cfg["backend"] == "gguf":
        return load_gguf_model(model_name)
    else:
        return load_hf_model(model_name)


# Unified predict, prompt_data is list[dict] for HF, str for GGUF
def predict(model, tokenizer, prompt_data, model_name: str) -> int:
    cfg = MODELS[model_name]
    max_tokens = cfg.get("max_new_tokens", 6)

    if cfg["backend"] == "gguf":
        return predict_gguf(model, prompt_data, max_tokens)
    else:
        enable_thinking = cfg.get("enable_thinking", True)
        return predict_hf(model, tokenizer, prompt_data, max_tokens, enable_thinking=enable_thinking)
