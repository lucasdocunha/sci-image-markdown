"""
Avaliação end-to-end do experimento B (M1 esqueleto + M2 valores -> M3 fusão).

Reporta:
- métricas finais (mesmo TableExtractionEvaluator do evaluate.py), sobre a saída do M3;
- as mesmas métricas para a fusão determinística (referência sem M3), para isolar o ganho do M3;
- métricas intermediárias: acerto de forma do M1 e acurácia de célula do M2.
"""

import json
import os
from typing import Any, Dict, List, Optional

import click

from src.data.decomposed import body_cells, parse_values, table_shapes, make_skeleton
from src.metrics.evaluator import TableExtractionEvaluator
from src.metrics.table_parser import sanitize_cell_text
from src.utils.config import load_config
from src.utils.io import read_jsonl
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_decomposed")

DEFAULT_CFG = {
    "skeleton": "configs/training/train_teleocr_skeleton.yaml",
    "values": "configs/training/train_teleocr_values.yaml",
    "fusion": "configs/training/train_teleocr_fusion.yaml",
}


def _as_float(s: str) -> Optional[float]:
    try:
        return float(sanitize_cell_text(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _cell_match(pred: str, gt: str, rel_tol: float) -> bool:
    p, g = _as_float(pred), _as_float(gt)
    if p is not None and g is not None:
        return abs(p - g) <= rel_tol * max(abs(g), 1e-12)
    return pred.strip() == gt.strip()


def value_cell_accuracy(pred_values: str, gt_md: str, rel_tol: float) -> Optional[float]:
    """Fração das células de dados GT cuja célula prevista (mesma posição) bate.

    Usa a posição (tabela, linha, coluna) como chave; células faltantes contam como erro.
    """
    gt_tables = body_cells(gt_md)
    pr_tables = parse_values(pred_values)
    total = hits = 0
    for t_idx, gt_rows in enumerate(gt_tables):
        pr_rows = pr_tables[t_idx] if t_idx < len(pr_tables) else []
        for r_idx, gt_cells in enumerate(gt_rows):
            pr_cells = pr_rows[r_idx] if r_idx < len(pr_rows) else []
            for c_idx, gt_c in enumerate(gt_cells):
                total += 1
                if c_idx < len(pr_cells) and _cell_match(pr_cells[c_idx], gt_c, rel_tol):
                    hits += 1
    return (hits / total) if total else None


def _resolve_image(img_path: str, folders: List[str]) -> str:
    candidates = [os.path.join(f, img_path) for f in folders if f]
    candidates += [os.path.join(f, os.path.basename(img_path)) for f in folders if f]
    candidates += [img_path, os.path.join("data/processed", img_path), os.path.join("data/raw/images", os.path.basename(img_path))]
    return next((p for p in candidates if os.path.exists(p)), img_path)


@click.command()
@click.option("--config-skeleton", default=DEFAULT_CFG["skeleton"], show_default=True)
@click.option("--config-values", default=DEFAULT_CFG["values"], show_default=True)
@click.option("--config-fusion", default=DEFAULT_CFG["fusion"], show_default=True)
@click.option("--adapter-skeleton", default=None)
@click.option("--adapter-values", default=None)
@click.option("--adapter-fusion", default=None)
@click.option("--test-file", default="data/processed/test.jsonl", show_default=True)
@click.option("--output-report", required=True)
@click.option("--save-predictions", default=None)
@click.option("--max-samples", default=None, type=int)
def main(config_skeleton, config_values, config_fusion, adapter_skeleton, adapter_values, adapter_fusion,
         test_file, output_report, save_predictions, max_samples):
    import torch
    from tqdm import tqdm

    from src.inference.decomposed import DecomposedTablePipeline

    cfg_s, cfg_v, cfg_f = load_config(config_skeleton), load_config(config_values), load_config(config_fusion)
    rel_tol = cfg_s.get("evaluation", {}).get("numerical_relative_tolerance", 0.05)
    evaluator = TableExtractionEvaluator(rel_tol=rel_tol)

    records = read_jsonl(test_file)
    if max_samples:
        records = records[:max_samples]
    gts = [r["table"] for r in records]

    pipeline = DecomposedTablePipeline(
        cfg_s, cfg_v, cfg_f,
        adapters={"skeleton": adapter_skeleton, "values": adapter_values, "fusion": adapter_fusion},
    )
    folders = [cfg_s.get("data", {}).get("image_folder")]

    if save_predictions is None:
        save_predictions = os.path.join(os.path.dirname(os.path.abspath(output_report)), "predictions.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(save_predictions)), exist_ok=True)

    from PIL import Image

    items: List[Dict[str, Any]] = []
    with open(save_predictions, "w", encoding="utf-8") as fh, torch.inference_mode():
        for idx, rec in enumerate(tqdm(records, desc="Decomposed", unit="sample")):
            path = _resolve_image(rec["image"], folders)
            try:
                res = pipeline.predict_image(Image.open(path))
            except Exception as exc:  # keep going; a failed sample counts as empty
                logger.error(f"Sample {idx} ({rec['image']}) failed: {exc}")
                res = {"skeleton": "", "values": "", "deterministic_table": "", "markdown_table": "",
                       "fused_raw": "", "fallback_used": True}
            item = {"index": idx, "image": rec["image"], "ground_truth": gts[idx], **res}
            items.append(item)
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            fh.flush()

    finals = [i["markdown_table"] for i in items]
    dets = [i["deterministic_table"] for i in items]

    report: Dict[str, Any] = {}
    report.update(evaluator.evaluate_batch(finals, gts))
    det_metrics = evaluator.evaluate_batch(dets, gts)
    report.update({f"deterministic_{k}": v for k, v in det_metrics.items()})

    shape_hits = sum(1 for i in items if table_shapes(i["skeleton"]) == table_shapes(make_skeleton(i["ground_truth"])))
    report["skeleton_shape_match"] = shape_hits / len(items) if items else float("nan")

    accs = [value_cell_accuracy(i["values"], i["ground_truth"], rel_tol) for i in items]
    accs = [a for a in accs if a is not None]
    report["value_cell_accuracy"] = sum(accs) / len(accs) if accs else float("nan")
    report["fallback_rate"] = sum(1 for i in items if i["fallback_used"]) / len(items) if items else float("nan")
    report["num_samples"] = len(items)

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    for k, v in report.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    logger.info(f"Report -> {output_report} | predictions -> {save_predictions}")


if __name__ == "__main__":
    main()
