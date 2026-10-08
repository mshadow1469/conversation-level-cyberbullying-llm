# Conversation-Level Cyberbullying Detection with LLMs

Prompting strategies, metadata augmentation, and parameter-efficient fine-tuning (LoRA vs LoRMA+) for detecting cyberbullying across whole conversations rather than single messages.

BSc Computer Science dissertation (COM3001), University of Surrey, May 2026. Supervised by Diptesh Kanojia.
📄 **[Read the full report](docs/Manraj_Boparai_COM3001_Final_Report.pdf)**

---

## Overview

Most cyberbullying detectors classify individual messages. This project classifies a **full user-pair conversation**, using a dataset annotated for four aspects of cyberbullying: aggression, repetition, peerness, and intent to harm.

I evaluated four open-weight instruction-tuned LLMs:

| Model | Size |
|---|---|
| Llama 3.2-Instruct | 3B |
| Mistral-Instruct-v0.3 | 7B |
| Qwen 3.5 | 9B |
| Gemma 3-IT | 27B (4-bit GGUF) |

across **8 prompting configurations** (zero-shot, few-shot, and metadata-augmented variants using peerness, intent, and aggression), plus **3 fine-tuned configurations** on Llama 3.2-3B comparing LoRA against LoRMA+, a recent multiplicative low-rank adaptation method.

## Key results

Macro F1 on the held-out test set:

| Approach | Macro F1 |
|---|---|
| Best prompting (Gemma 3-27B, zero-shot + all metadata) | 0.6263 |
| LoRMA+ (reference-implementation init) | 0.5564 |
| LoRMA+ (paper init) | 0.7285 |
| **LoRA** | **0.7437** |

Main findings:

- **All fine-tuned configs beat the best prompting result**, with the exception of LoRMA+ with the reference-implementation init (0.5564).
- **Initialisation matters a lot.** I found a discrepancy between the LoRMA+ paper and its official code. Under otherwise identical conditions the two initialisations differ by **0.1721 macro F1**.
- **Prompting over-predicts the minority class.** Models frequently flag non-bullying conversations as bullying (high recall, low precision).
- **Few-shot did not reliably help**, and often hurt, for most models.

Full per-configuration results (precision, recall, F1, confusion matrices) are in [`results_summary.txt`](results_summary.txt) and Appendix A.4 of the report.

## Repository structure

```text
.
├── run_experiment.py          # Unified CLI for all prompting experiments
├── requirements.txt
├── results_summary.txt        # Aggregated results table
├── src/
│   ├── framework/
│   │   ├── config.py          # Model registry, paths, strategy names
│   │   ├── data.py            # CSV loading, conversation parsing, metadata
│   │   ├── prompts.py         # Base prompt + strategy dispatch
│   │   ├── few_shot_examples.py
│   │   ├── models.py          # HF Transformers + llama.cpp backends
│   │   └── evaluate.py        # Macro F1, per-class metrics, confusion matrix
│   └── finetuning/
│       ├── lora/              # LoRA baseline
│       └── lorma/             # LoRMA+ (two initialisation variants)
├── baselines/
│   └── tfidf_lr.py            # TF-IDF + logistic regression baseline
├── legacy/                    # Original per-script versions (pre-refactor)
├── data_processed/            # Pre-split train/val/test CSVs
└── docs/
    └── Manraj_Boparai_COM3001_Final_Report.pdf
```

## Setup

Tested on Python 3.12, Linux, CUDA 13.0, NVIDIA RTX 4000 Ada (20 GB VRAM).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the Gemma 3-27B backend you also need `llama-cpp-python` and the quantised model file:

```bash
pip install llama-cpp-python
```

Download `gemma-3-27b-it.IQ4_XS.gguf` (~14 GB) from [mradermacher/gemma-3-27b-it-GGUF](https://huggingface.co/mradermacher/gemma-3-27b-it-GGUF).

### Configure paths

The code was developed on University of Surrey lab machines, so cache and model paths are hardcoded. Before running, edit the paths in:

- `src/framework/config.py`
- `src/finetuning/lora/llama_lora.py`
- `src/finetuning/lorma/llama_lorma_plus_akaiming_b0.py`
- `src/finetuning/lorma/llama_lorma_plus_a0_bkaiming.py`

## Usage

### Prompting experiments

```bash
python run_experiment.py --model llama --strategy zero_shot
python run_experiment.py --model gemma --strategy zero_shot_meta
python run_experiment.py --model all   --strategy zero_shot
python run_experiment.py --model llama --strategy all
```

**Models:** `llama`, `gemma`, `qwen`, `mistral`
**Strategies:** `zero_shot`, `few_shot`, `zero_shot_peer`, `few_shot_peer`, `zero_shot_intent`, `zero_shot_aggressive`, `zero_shot_meta`, `few_shot_meta`

Results are written to `outputs/results/`.

### Classical baseline

```bash
python -m baselines.tfidf_lr
```

### Fine-tuning

```bash
python -m src.finetuning.lora.llama_lora                           # LoRA
python -m src.finetuning.lorma.llama_lorma_plus_akaiming_b0        # LoRMA+ (paper init)
python -m src.finetuning.lorma.llama_lorma_plus_a0_bkaiming        # LoRMA+ (reference-code init)
```

Each run takes roughly 3 to 6 hours on a single 20 GB GPU. Adapters are saved to `outputs/<run_name>/`.

## Data

`data_processed/` contains pre-split train/val/test CSVs derived from [`surrey-nlp/Cyberbullying-Detection-CB2`](https://huggingface.co/datasets/surrey-nlp/Cyberbullying-Detection-CB2).

> ⚠️ The data contains abusive and offensive language.

## Notes on the LoRMA+ implementation

LoRMA+ is implemented as a Hugging Face-compatible wrapper around the attention projection layers. Both initialisation schemes are included so the paper-vs-code discrepancy can be reproduced; see Section 4 of the report for the analysis.

## Citation

If you use this work, please cite:

```bibtex
@thesis{boparai2026cyberbullying,
  author = {Boparai, Manraj},
  title  = {Conversation-Level Cyberbullying Detection Using Large Language Models},
  school = {University of Surrey},
  year   = {2026},
  type   = {BSc dissertation}
}
```

## License

MIT (code). The dataset retains its original license; see the source link above.
