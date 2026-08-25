# ADR 0001: Modular VLM Fine-Tuning and Hybrid Evaluation Pipeline for Sci-ImageMiner Task 2

## Status
Accepted

## Context
The ICDAR 2026 Sci-ImageMiner Task 2 requires converting scientific figure panels (plots, graphs, spectra) into structured Markdown data tables. The project requires an extensible, reproducible, and modular framework capable of fine-tuning local open-source Vision-Language Models (such as Qwen2.5-VL, InternVL2, Florence-2) while accurately evaluating both the syntactic structural integrity and numerical precision of extracted tables.

## Decision
1. **Model Stack**:
   - Use Hugging Face ecosystem (`transformers`, `peft`, `trl`, `accelerate`).
   - Standardize parameter-efficient fine-tuning via QLoRA (4-bit `bitsandbytes`) and LoRA (`bfloat16`/`fp16`).
   - Support modular model adapters starting with Qwen2.5-VL (3B/7B) and extensible to Florence-2 / InternVL.

2. **Evaluation Protocol**:
   - Implement a hybrid evaluation engine that does not rely solely on naive text generation metrics (e.g. BLEU/ROUGE).
   - Parse generated Markdown into structured DataFrames.
   - Compute:
     - **Valid Table Rate (VTR)**: Structural validity.
     - **Cell-Level Numerical Metrics**: Root Mean Squared Error (RMSE) and Relative Numerical Error (RNE).
     - **Structural Similarity**: Normalized Table Edit Distance / TEDS.
     - **Lexical Overlap**: ROUGE-L and BLEU-4.

3. **Project Architecture**:
   - Config-driven YAML architecture (`configs/`) with clean separation of concerns in `src/` (`src/data`, `src/models`, `src/training`, `src/metrics`, `src/inference`, `src/utils`).
   - Clear CLI entrypoints (`prepare_data.py`, `train.py`, `evaluate.py`, `predict.py`).

## Consequences
- **Positive**:
  - Reproducible experiments easily managed via YAML configurations.
  - High flexibility in switching model architectures and training regimes (QLoRA vs LoRA).
  - Domain-specific evaluation that measures true scientific extraction accuracy.
- **Negative**:
  - Requires maintaining custom table parsers for diverse Markdown table variants.
