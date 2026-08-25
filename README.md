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
python evaluate.py --config configs/default.yaml --test-file data/processed/test.jsonl
```

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
