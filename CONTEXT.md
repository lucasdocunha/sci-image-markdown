# Domain Context: Sci-ImageMiner Task 2 (Data to Markdown)

This document defines the core domain concepts, entities, and research context for scientific figure table extraction.

## 1. Domain Overview

The **Sci-ImageMiner** benchmark (associated with **ICDAR 2026**) focuses on multimodal understanding of authentic scientific figures from the materials science domain, specifically **Atomic Layer Deposition and Etching (ALD/E)**.

Scientific figures in this domain (growth-per-cycle curves, saturation curves, spectroscopic ellipsometry, X-ray photoelectron spectroscopy, temperature windows) contain high-density quantitative data essential for reproducing experiments.

### Task 2: Data Table Extraction (Data to Markdown)
The goal of Task 2 is to take an input image containing one or more scientific figure panels / plots and reconstruct the underlying numerical and tabular data as a structured **Markdown table**.

## 2. Glossary & Core Terminology

- **Figure Panel**: An individual plot or subplot within a scientific figure (e.g., panel (a), panel (b)).
- **Data Table / Ground Truth Table**: The original tabular data points plotted in the figure, formatted as a standard GitHub-Flavored Markdown table with column headers and numerical rows.
- **VLM (Vision-Language Model)**: Multimodal models (such as `Qwen2.5-VL`, `InternVL2`, `Florence-2`) that accept both visual tokens (images) and text prompts to generate structured textual output.
- **QLoRA (Quantized Low-Rank Adaptation)**: Parameter-efficient fine-tuning technique that freezes a 4-bit quantized base model and attaches trainable low-rank adapter matrices.
- **Table Parsing**: The process of taking raw generated Markdown text, validating its syntax, and converting it into a structured DataFrame (`pandas.DataFrame`) for row/column alignment.
- **Relative Numerical Error (RNE)**: The relative percentage error between extracted numerical cell values and ground truth cell values ($|y_{pred} - y_{true}| / (|y_{true}| + \epsilon)$).
- **Tree Edit Distance / Table Edit Distance (TEDS)**: Structural and content metric measuring the tree edit distance between parsed table trees.
- **Valid Table Rate (VTR)**: The percentage of model generations that can be successfully parsed into a non-empty, well-formed table.

## 3. Data Structure

```
data/
├── raw/                 # Unprocessed original annotations and figures
│   ├── train.jsonl / train_annotations.json
│   ├── val.jsonl
│   └── images/
├── processed/           # Formatted conversations and standardized image-table pairs
└── splits/              # Stratified train/val/test splits
```

## 4. Key Modeling Guidelines

- Prompt formatting must clearly instruct the VLM to extract the quantitative data points from the visual plot into a Markdown table with proper column headers.
- When evaluating, always separate structural validity (is it a valid Markdown table?) from numerical precision (are the coordinates / numbers accurate within an acceptable error tolerance?).
