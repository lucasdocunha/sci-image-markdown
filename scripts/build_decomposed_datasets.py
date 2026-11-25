"""
Gera os JSONL do experimento B a partir dos splits existentes (train/val/test).

Saídas (em --processed-dir):
  decomp_skeleton_{train,val,test}.jsonl   M1: imagem -> esqueleto
  decomp_values_{train,val,test}.jsonl     M2: imagem -> valores (sem cabeçalho)
  decomp_fusion_{train,val}.jsonl          M3 (--fusion-source gt): esqueleto+valores GT -> tabela GT
  decomp_fusion_train_pred.jsonl           M3 (--fusion-source pred): esqueleto+valores PREVISTOS -> GT

Para --fusion-source pred, passe --pred-file com as predições de evaluate_decomposed.py
geradas num split que M1/M2 não viram no treino (ex.: val), para treinar M3 com erros reais.
"""

import os

import click

from src.data.decomposed import make_skeleton, make_values
from src.utils.io import read_jsonl, write_jsonl

SPLITS = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}


@click.command()
@click.option("--processed-dir", default="data/processed", show_default=True)
@click.option("--fusion-source", type=click.Choice(["gt", "pred"]), default="gt", show_default=True)
@click.option("--pred-file", default=None, help="Predições de evaluate_decomposed.py (necessário com --fusion-source pred).")
def main(processed_dir: str, fusion_source: str, pred_file: str):
    for split, fname in SPLITS.items():
        recs = read_jsonl(os.path.join(processed_dir, fname))

        skel = [{"id": r["id"], "image": r["image"], "table": make_skeleton(r["table"])} for r in recs]
        vals = [{"id": r["id"], "image": r["image"], "table": make_values(r["table"])} for r in recs]
        write_jsonl(skel, os.path.join(processed_dir, f"decomp_skeleton_{split}.jsonl"))
        write_jsonl(vals, os.path.join(processed_dir, f"decomp_values_{split}.jsonl"))
        click.echo(f"[{split}] {len(recs)} samples -> skeleton + values")

        if fusion_source == "gt" and split in ("train", "val"):
            fus = [
                {
                    "id": r["id"],
                    "skeleton": make_skeleton(r["table"]),
                    "values": make_values(r["table"]),
                    "table": r["table"],
                }
                for r in recs
            ]
            write_jsonl(fus, os.path.join(processed_dir, f"decomp_fusion_{split}.jsonl"))
            click.echo(f"[{split}] {len(fus)} fusion samples (GT inputs)")

    if fusion_source == "pred":
        if not pred_file:
            raise click.UsageError("--fusion-source pred requer --pred-file")
        preds = read_jsonl(pred_file)
        fus = [
            {
                "id": os.path.basename(str(p.get("image", p.get("index")))),
                "skeleton": p["skeleton"],
                "values": p["values"],
                "table": p["ground_truth"],
            }
            for p in preds
        ]
        out = os.path.join(processed_dir, "decomp_fusion_train_pred.jsonl")
        write_jsonl(fus, out)
        click.echo(f"{len(fus)} fusion samples (predicted inputs) -> {out}")


if __name__ == "__main__":
    main()
