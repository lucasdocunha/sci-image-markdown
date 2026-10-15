"""
Measures how well each rule matches the manual review, to tune the thresholds.
"""

import pandas as pd

from .taxonomy import CATEGORIES

REVIEWED = {"sim", "não", "nao"}


def _labels(value) -> set:
    return {c.strip() for c in str(value).split(";") if c.strip()}


def rule_precision(panels_csv: str, review_csv: str) -> pd.DataFrame:
    """Per category (and gt_suspeito): tp, fp, fn, precision and recall of the rules.

    Only review rows whose gt_suspeito was filled in (sim/não) count as reviewed.
    """
    panels = pd.read_csv(panels_csv, keep_default_na=False)
    review = pd.read_csv(review_csv, keep_default_na=False)
    review = review[review["gt_suspeito"].str.strip().str.lower().isin(REVIEWED)]
    keys = ["sample_id", "panel_id"]
    df = review[keys + ["categorias", "gt_suspeito"]].merge(
        panels[keys + ["categorias", "gt_suspeito"]], on=keys, suffixes=("_manual", "_regra")
    )

    rule_sets = df["categorias_regra"].map(_labels)
    manual_sets = df["categorias_manual"].map(_labels)
    # Explicit bool dtype: on an empty review, map() would return a str-typed Series.
    hits = {
        cat: (
            pd.Series([cat in s for s in rule_sets], dtype=bool),
            pd.Series([cat in s for s in manual_sets], dtype=bool),
        )
        for cat in CATEGORIES
    }
    hits["gt_suspeito"] = (
        pd.Series(df["gt_suspeito_regra"].astype(str).eq("True").tolist(), dtype=bool),
        pd.Series(df["gt_suspeito_manual"].astype(str).str.strip().str.lower().eq("sim").tolist(), dtype=bool),
    )

    rows = []
    for cat, (rule, manual) in hits.items():
        tp, fp, fn = int((rule & manual).sum()), int((rule & ~manual).sum()), int((~rule & manual).sum())
        rows.append({
            "categoria": cat, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        })
    return pd.DataFrame(rows)
