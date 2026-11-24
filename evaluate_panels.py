"""
CLI tool for per-panel inference: crops every panel by its metadata bbox, predicts each
crop alone and writes the predictions in the official submission format.

Score the output with: analyze_errors.py --official-predictions <out-dir>/predictions_official.jsonl
"""

import json
import os

import click
from PIL import Image

from src.inference.panels import predict_panels
from src.utils.config import load_config, merge_configs
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_panels")


def _has_ground_truth(record: dict) -> bool:
    return any((d.get("data") or "").strip() for d in (record.get("data_extraction") or []))


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to base configuration YAML.")
@click.option("--model-config", default=None, help="Optional model override configuration YAML.")
@click.option("--training-config", default=None, help="Optional training/data override configuration YAML.")
@click.option("--metadata", default="data/raw/validation_metadata.jsonl", help="HF metadata.jsonl with the bbox of every panel.")
@click.option("--image-root", default="data/raw", help="Folder the metadata file_name is relative to.")
@click.option("--adapter-path", default=None, help="Optional path to trained PEFT LoRA adapter.")
@click.option("--margin", default=0.0, type=float, help="Fraction of the bbox size added around each crop.")
@click.option("--out-dir", required=True, help="Directory for panel_predictions.jsonl and predictions_official.jsonl.")
@click.option("--max-samples", default=None, type=int, help="Optional maximum number of figures.")
def main(config, model_config, training_config, metadata, image_root, adapter_path, margin, out_dir, max_samples):
    """Predicts one table per panel crop for every figure with ground truth."""
    import torch
    from tqdm import tqdm

    from src.inference.predictor import TablePredictor

    cfg = load_config(config)
    for override in (model_config, training_config):
        if override:
            cfg = merge_configs(cfg, load_config(override))

    records = [r for r in read_jsonl(metadata) if _has_ground_truth(r) and r.get("bbox")]
    if max_samples:
        records = records[:max_samples]

    os.makedirs(out_dir, exist_ok=True)
    panel_file = os.path.join(out_dir, "panel_predictions.jsonl")
    # Appending lets an interrupted run resume where it stopped.
    done = read_jsonl(panel_file) if os.path.exists(panel_file) else []
    done_ids = {d["sample_id"] for d in done}
    logger.info(f"{len(records)} figures, {len(done_ids)} already predicted")

    predictor = TablePredictor(cfg, adapter_path=adapter_path)

    def predict(crop):
        return predictor.predict_image(crop)["raw_output"]

    with open(panel_file, "a", encoding="utf-8") as handle, torch.inference_mode():
        for rec in tqdm(records, unit="figure"):
            if rec["sample_id"] in done_ids:
                continue
            image = Image.open(os.path.join(image_root, rec["file_name"])).convert("RGB")
            item = {"sample_id": rec["sample_id"], "panels": predict_panels(predict, image, rec["bbox"], margin)}
            done.append(item)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    official = [
        {
            "sample_id": d["sample_id"],
            "data_extraction": {pid: p["table"] for pid, p in d["panels"].items() if p["table"]},
        }
        for d in done
    ]
    write_jsonl(official, os.path.join(out_dir, "predictions_official.jsonl"))
    extra = sum(p["extra_tables"] for d in done for p in d["panels"].values())
    logger.info(f"Wrote {len(official)} figures to {out_dir} ({extra} extra tables dropped)")


if __name__ == "__main__":
    main()
