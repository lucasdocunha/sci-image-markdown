"""
CLI tool for benchmarking and evaluating predictions against ground truth tables.
"""

import os
import json
import click
from rich.table import Table
from rich.console import Console

from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger
from src.utils.io import read_jsonl
from src.metrics.evaluator import TableExtractionEvaluator
from src.inference.predictor import TablePredictor

logger = setup_logger("evaluate")
console = Console()


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to base configuration YAML.")
@click.option("--test-file", default="data/processed/test.jsonl", help="Path to test dataset JSONL.")
@click.option("--adapter-path", default=None, help="Optional path to trained PEFT LoRA adapter.")
@click.option("--predictions-file", default=None, help="Optional path to pre-generated predictions JSONL for offline evaluation.")
@click.option("--output-report", default="outputs/eval_results/report.json", help="Path to save evaluation JSON report.")
@click.option("--save-predictions", default=None, help="Optional path to save generated predictions JSONL.")
def main(
    config: str,
    test_file: str,
    adapter_path: str,
    predictions_file: str,
    output_report: str,
    save_predictions: str,
):
    """Runs full evaluation and prints domain table extraction metrics."""
    from tqdm import tqdm
    import torch

    cfg = load_config(config)
    eval_cfg = cfg.get("evaluation", {})
    evaluator = TableExtractionEvaluator(rel_tol=eval_cfg.get("numerical_relative_tolerance", 0.05))

    logger.info(f"Loading test records from {test_file}...")
    test_records = read_jsonl(test_file)
    ground_truths = [r.get("table", r.get("markdown", "")) for r in test_records]

    pred_records = []
    if predictions_file:
        logger.info(f"Loading predictions from existing file: {predictions_file}")
        pred_records = read_jsonl(predictions_file)
        predictions = [p.get("markdown_table", p.get("prediction", p.get("raw_output", ""))) for p in pred_records]
    else:
        tag = "Fine-Tuned Adapter" if adapter_path else "Base Model"
        logger.info(f"Instantiating TablePredictor for evaluation ({tag})...")
        predictor = TablePredictor(cfg, adapter_path=adapter_path)
        predictions = []
        img_folder = cfg.get("data", {}).get("image_folder", "data/raw/images")
        
        with torch.inference_mode():
            for idx, rec in enumerate(tqdm(test_records, desc=f"Evaluating [{tag}]", unit="sample")):
                img_path = rec["image"]
                candidates = []
                if os.path.isabs(img_path):
                    candidates.append(img_path)
                if img_folder:
                    candidates.append(os.path.join(img_folder, img_path))
                    candidates.append(os.path.join(img_folder, os.path.basename(img_path)))
                    candidates.append(os.path.join(os.path.dirname(img_folder), img_path))
                candidates.append(os.path.join("data/raw/images", os.path.basename(img_path)))
                candidates.append(os.path.join("data/raw", img_path))
                candidates.append(os.path.join("data/processed_smoke/images", os.path.basename(img_path)))
                candidates.append(os.path.join("data", img_path))
                candidates.append(img_path)
                resolved_path = next((p for p in candidates if os.path.exists(p)), None)
                if not resolved_path:
                    logger.warning(f"Could not resolve image path for {img_path}. Tried: {candidates[:3]}")
                    resolved_path = img_path

                try:
                    res = predictor.predict_image(resolved_path)
                    pred_table = res["markdown_table"]
                    is_valid = res["is_valid_table"]
                except Exception as exc:
                    logger.error(f"Error predicting sample {idx} ({img_path}): {exc}")
                    pred_table = ""
                    is_valid = False

                predictions.append(pred_table)
                pred_records.append({
                    "index": idx,
                    "image": img_path,
                    "ground_truth": ground_truths[idx],
                    "markdown_table": pred_table,
                    "is_valid_table": is_valid,
                })

                if torch.cuda.is_available() and (idx + 1) % 25 == 0:
                    torch.cuda.empty_cache()

    logger.info(f"Evaluating metrics over {len(predictions)} samples...")
    results = evaluator.evaluate_batch(predictions, ground_truths)

    # Print nicely formatted summary table
    report_tag = "Fine-Tuned Adapter" if adapter_path else "Base Model (Zero-Shot)"
    table = Table(title=f"Evaluation Report: {report_tag}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for k, v in results.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}" if not (v != v) else "NaN")
        else:
            table.add_row(k, str(v))

    console.print(table)

    # Save output report
    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation report saved to: {output_report}")

    # Save predictions jsonl
    if save_predictions is None:
        save_predictions = os.path.join(os.path.dirname(os.path.abspath(output_report)), "predictions.jsonl")
    
    if pred_records:
        with open(save_predictions, "w", encoding="utf-8") as f:
            for item in pred_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Sample predictions saved to: {save_predictions}")


if __name__ == "__main__":
    main()
