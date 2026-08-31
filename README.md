# sci-image-markdown

Research framework for **Scientific Figure & Plot Extraction to Structured Markdown Tables** (Sci-ImageMiner / ICDAR 2026 Task 2: Data Table Extraction).

---

## 📌 Overview

Scientific figure panels (ALD/E saturation curves, growth rate windows, spectroscopic ellipsometry, XPS spectra) contain critical quantitative measurements. `sci-image-markdown` provides an end-to-end framework to:

1. **Ingest & Structure** scientific figure image-table pairs.
2. **Fine-tune Open-Source VLMs** (e.g. Qwen2.5-VL 3B/7B, Florence-2) with **QLoRA** (4-bit NF4) or **LoRA** (BF16/FP16).
3. **Evaluate with Domain Metrics**: Valid Table Rate (VTR), Relative Numerical Error (RNE), RMSE, Normalized Table Edit Distance, Lexical Overlap (ROUGE-L, BLEU-4), and cell-level Precision / Recall / F1.
4. **Run Inference & CLI Batch Predictions** converting figures directly into Markdown and Pandas CSV tables.

---

## 🏗️ Repository Architecture

```
sci-image-markdown/
├── AGENTS.md                  # Agent workflows and skill guidelines
├── CONTEXT.md                 # Domain model, glossary, and research concepts
├── configs/                   # Modular YAML experiment configs
│   ├── default.yaml           # Base parameters
│   ├── models/                # Architecture configs (Qwen2.5-VL 3B/7B, Florence-2)
│   └── training/              # Optimization configs (QLoRA, LoRA)
├── docs/
│   ├── adr/                   # Architecture Decision Records
│   └── agents/                # Triage, domain, and issue tracker specs
├── src/
│   ├── data/                  # Datasets, preprocessors, and collators
│   ├── models/                # Model loaders and architecture wrappers
│   ├── metrics/               # Table parsers, RNE, RMSE, edit distance, evaluator
│   ├── training/              # SFT Trainer and training loops
│   ├── inference/             # Single-image & batch predictors
│   └── utils/                 # Logging, I/O, and config mergers
├── prepare_data.py            # CLI: dataset split generation & synthetic demo data
├── train.py                   # CLI: model fine-tuning entrypoint
├── evaluate.py                # CLI: benchmark evaluation & rich reporting
├── predict.py                 # CLI: single-image extraction to Markdown / CSV
└── tests/                     # Automated unit and integration tests
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -e .
```

### 2. Prepare Data

```bash
# Generate synthetic verification samples for quick testing:
python prepare_data.py --create-synthetic --num-samples 20

# Or process raw Sci-ImageMiner data:
python prepare_data.py --raw-dir data/raw --processed-dir data/processed
```

### 3. Fine-Tune VLM (QLoRA)

```bash
# Train Qwen2.5-VL 3B with 4-bit QLoRA
python train.py --config configs/default.yaml

# Train Qwen2.5-VL 7B with custom override
python train.py --config configs/default.yaml --model-config configs/models/qwen2_5_vl_7b.yaml
```

### 4. Evaluate Benchmark

```bash
# Run full evaluation comparing Base Model vs Fine-Tuned LoRA:
bash run_eval_both.sh
```

---

## 📊 Benchmark & SOTA Comparison

Evaluated on **373 independent test figure images** from the **Sci-ImageMiner** benchmark (ICDAR 2026 Task 2):

| Metric / Capability | Base Model (Zero-Shot) | Fine-Tuned (LoRA 3B) | SOTA (VLMinators / GPT-4o) | Delta vs Base |
| :--- | :---: | :---: | :---: | :---: |
| **ICDAR Task 2 Final Score** | 31.08 | **`40.50`** | `40.80` (VLMinators 7B) | **+9.42 pts** 🎯 |
| **Valid Table Rate (VTR)** | 88.74% | **99.73%** | ~98.8% (VLMinators) | **+10.99%** 🚀 |
| **Cell Recall ($\le 5\%$ tol.)** | 35.78% | **48.22%** | ~42.0% - 49.0% | **+12.45%** 🚀 |
| **Cell RMSE** | 2.6635 | **1.0308** | ~1.10 - 2.50 | **-1.6327 (-61%)** 🎯 |
| **Cell RNE (Relative Error)** | 0.0044 | **0.0036** | ~0.0040 - 0.0070 | **-0.0008** 🎯 |
| **Execution Mode** | Local (3B) | **Local (3B, < 8GB VRAM)** | Multi-GPU / Cloud API | **100% On-Premise** 🔒 |

👉 See the complete report, metric explanations, SOTA analysis, and qualitative comparisons in [BENCHMARK_RESULTS.md](file:///home/lucas/sci-image-markdown/BENCHMARK_RESULTS.md).

---

### 5. Run Single Image Inference

```bash
python predict.py --image path/to/figure.png --output-md table.md --output-csv table.csv
```

---

## 🧪 Testing

Run the test suite with pytest:

```bash
pytest -v
```
