# src/framework/evaluate.py
"""
Evaluation metrics and results saving.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


# Computes all evaluation metrics
def evaluate_predictions(labels: list[int], preds: list[int]) -> dict:
    report = classification_report(
        labels, preds, digits=4, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(labels, preds).tolist()
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "macro_f1": macro_f1,
        "accuracy": report["accuracy"],
        "class_0": report.get("0", {}),
        "class_1": report.get("1", {}),
        "confusion_matrix": cm,
        "n_samples": len(labels),
        "n_positive": sum(labels),
        "n_predicted_positive": sum(preds),
    }


# Prints results to console
def print_results(results: dict) -> None:
    print(f"\nMacro F1: {results['macro_f1']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Class 1 precision: {results['class_1'].get('precision', 0):.4f}")
    print(f"Class 1 recall:    {results['class_1'].get('recall', 0):.4f}")
    print(f"Class 1 F1:        {results['class_1'].get('f1-score', 0):.4f}")
    print(f"Confusion Matrix:  {results['confusion_matrix']}")
    print(f"Predicted positive: {results['n_predicted_positive']} / {results['n_samples']}")


# Saves results to a JSON file
def save_results(
    results: dict,
    model_name: str,
    strategy: str,
    output_dir: Path,
    preds: list[int] | None = None,
    labels: list[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "model": model_name,
        "strategy": strategy,
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
    }

    if preds is not None:
        output["predictions"] = preds
    if labels is not None:
        output["labels"] = labels

    filename = f"{model_name}_{strategy}.json"
    path = output_dir / filename

    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {path}")
