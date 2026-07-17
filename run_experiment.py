#!/usr/bin/env python3
# run_experiment.py
"""
Unified runner for  experiments.
Usage:
    python run_experiment.py --model llama --strategy zero_shot
    python run_experiment.py --model llama --strategy all
    python run_experiment.py --model all --strategy zero_shot

Available models:   llama, gemma, qwen, mistral
Available strategies: zero_shot, few_shot, zero_shot_peer, few_shot_peer,
                      zero_shot_intent, zero_shot_aggressive,
                      zero_shot_meta, few_shot_meta
"""
from __future__ import annotations

import argparse
import gc
import torch
from tqdm import tqdm

from src.framework.config import MODELS, VALID_STRATEGIES, OUTPUT_DIR
from src.framework.data import load_splits, extract_conversation_text, extract_metadata
from src.framework.prompts import build_messages, build_text_prompt
from src.framework.models import load_model, predict
from src.framework.evaluate import evaluate_predictions, print_results, save_results


# Run one model × one strategy and return the results dict
def run_single_experiment(model_name: str, strategy: str, save: bool = True) -> dict:
    print("=" * 70)
    print(f"  Model: {model_name}  |  Strategy: {strategy}")
    print("=" * 70)

    _, _, test_df = load_splits()
    print(f"Test size: {len(test_df)}")

    model, tokenizer = load_model(model_name)

    cfg = MODELS[model_name]
    is_gguf = cfg["backend"] == "gguf"
    max_chars = cfg.get("max_conv_chars", None)
    max_example_chars = cfg.get("max_example_chars", None)

    preds = []
    labels = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        conversation_text = extract_conversation_text(row, **({} if max_chars is None else {"max_chars": max_chars}))
        metadata = extract_metadata(row)

        if is_gguf:
            prompt_data = build_text_prompt(conversation_text, strategy, metadata, max_example_chars=max_example_chars)
        else:
            prompt_data = build_messages(conversation_text, strategy, metadata)

        pred = predict(model, tokenizer, prompt_data, model_name)
        preds.append(pred)
        labels.append(int(row["label"]))

    results = evaluate_predictions(labels, preds)
    print_results(results)

    if save:
        save_results(
            results, model_name, strategy,
            OUTPUT_DIR / "results",
            preds=preds, labels=labels,
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run cyberbullying detection experiments")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--strategy", type=str, required=True, choices=VALID_STRATEGIES + ["all"])
    parser.add_argument("--no-save", action="store_true", help="Do not save results to disk")
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    strategies = VALID_STRATEGIES if args.strategy == "all" else [args.strategy]

    all_results = {}

    for model_name in models:
        for strategy in strategies:
            key = f"{model_name}_{strategy}"
            try:
                results = run_single_experiment(model_name, strategy, save=not args.no_save)
                all_results[key] = results
            except Exception as e:
                print(f"ERROR in {key}: {e}")
                all_results[key] = {"error": str(e)}
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    # Prints summary table for experiments
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"{'Experiment':<35} {'Macro F1':>10} {'P(1)':>8} {'R(1)':>8}")
        print("-" * 65)
        for key, res in all_results.items():
            if "error" in res:
                print(f"{key:<35} {'ERROR':>10}")
            else:
                p1 = res["class_1"].get("precision", 0)
                r1 = res["class_1"].get("recall", 0)
                print(f"{key:<35} {res['macro_f1']:>10.4f} {p1:>8.4f} {r1:>8.4f}")


if __name__ == "__main__":
    main()
