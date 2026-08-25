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
def main(
    config: str,
    test_file: str,
    adapter_path: str,
    predictions_file: str,
    output_report: str,
):
    """Runs full evaluation and prints domain table extraction metrics."""
    cfg = load_config(config)
    eval_cfg = cfg.get("evaluation", {})
    evaluator = TableExtractionEvaluator(rel_tol=eval_cfg.get("numerical_relative_tolerance", 0.05))

    logger.info(f"Loading test records from {test_file}...")
    test_records = read_jsonl(test_file)
    ground_truths = [r.get("table", r.get("markdown", "")) for r in test_records]

    if predictions_file:
        logger.info(f"Loading predictions from existing file: {predictions_file}")
        pred_records = read_jsonl(predictions_file)
        predictions = [p.get("markdown_table", p.get("prediction", p.get("raw_output", ""))) for p in pred_records]
    else:
        logger.info("Instantiating TablePredictor for evaluation...")
        predictor = TablePredictor(cfg, adapter_path=adapter_path)
        predictions = []
        for idx, rec in enumerate(test_records):
            img_path = rec["image"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(cfg.get("data", {}).get("image_folder", "data/raw/images"), img_path)
            res = predictor.predict_image(img_path)
            predictions.append(res["markdown_table"])

    logger.info(f"Evaluating {len(predictions)} samples...")
    results = evaluator.evaluate_batch(predictions, ground_truths)

    # Print nicely formatted summary table
    table = Table(title="Sci-Image Task 2 (Data to Markdown) Evaluation Report", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for k, v in results.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}" if not (v != v) else "NaN")
        else:
            table.add_row(k, str(v))

    console.print(table)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation report saved to: {output_report}")


if __name__ == "__main__":
    main()
