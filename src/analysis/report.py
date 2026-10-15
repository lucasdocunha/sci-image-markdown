"""
Aggregations of panels.csv: how many score points each error category costs, and
scores broken down by a column.
"""

import pandas as pd

from .taxonomy import CATEGORIES

SCORE_COLUMNS = ["weighted", "rms", "teds"]


def category_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Per category: panels hit and points of the mean score lost to them.

    custo_total counts every panel in the category; custo_exclusivo only the
    panels where it is the single category, so those points are not shared.
    """
    labels = df["categorias"].fillna("").map(lambda s: [c for c in s.split(";") if c])
    loss = 100 - df["weighted"]
    total = len(df)
    rows = []
    for cat in CATEGORIES:
        hit = labels.map(lambda ls: cat in ls)
        alone = labels.map(lambda ls: ls == [cat])
        rows.append({
            "categoria": cat,
            "n": int(hit.sum()),
            "custo_total": float(loss[hit].sum() / total),
            "custo_exclusivo": float(loss[alone].sum() / total),
        })
    return pd.DataFrame(rows).sort_values("custo_total", ascending=False, kind="stable").reset_index(drop=True)


def slice_scores(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean scores per group of `by`, with the panel count, largest groups first."""
    cols = [c for c in SCORE_COLUMNS if c in df.columns]
    out = df.groupby(by)[cols].mean()
    out.insert(0, "n", df.groupby(by).size())
    return out.sort_values("n", ascending=False, kind="stable").reset_index()
