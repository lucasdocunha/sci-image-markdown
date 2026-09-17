"""
Domain metrics for Table Extraction: RNE, RMSE, Normalized Edit Distance, Precision, Recall, F1.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import Levenshtein


def compute_normalized_edit_distance(pred_str: str, target_str: str) -> float:
    """Computes normalized Levenshtein edit distance between two strings [0, 1].
    
    1.0 means identical, 0.0 means completely different.
    """
    if not pred_str and not target_str:
        return 1.0
    if not pred_str or not target_str:
        return 0.0

    dist = Levenshtein.distance(pred_str, target_str)
    max_len = max(len(pred_str), len(target_str))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - (dist / max_len))


def extract_numerical_cells(df: pd.DataFrame) -> List[float]:
    """Extracts all numeric float values present in a DataFrame."""
    values = []
    for col in df.columns:
        for val in df[col]:
            try:
                num = float(val)
                if not (math.isnan(num) or math.isinf(num)):
                    values.append(num)
            except (ValueError, TypeError):
                continue
    return values


def compute_numerical_cell_metrics(
    pred_df: Optional[pd.DataFrame],
    target_df: Optional[pd.DataFrame],
    rel_tol: float = 0.05
) -> Dict[str, float]:
    """Computes numerical cell precision, recall, F1, RMSE, and RNE.
    
    Matches extracted values against ground-truth values within `rel_tol` (default 5%).
    """
    if pred_df is None or target_df is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rmse": float("nan"),
            "mean_rne": float("nan"),
        }

    pred_vals = extract_numerical_cells(pred_df)
    target_vals = extract_numerical_cells(target_df)

    if not target_vals:
        return {
            "precision": 1.0 if not pred_vals else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_vals else 0.0,
            "rmse": 0.0 if not pred_vals else float("nan"),
            "mean_rne": 0.0 if not pred_vals else float("nan"),
        }

    if not pred_vals:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rmse": float("nan"),
            "mean_rne": float("nan"),
        }

    matched_targets = set()
    matched_preds = set()
    paired_errors = []
    paired_rnes = []

    # Greedy bipartite matching based on minimum relative difference
    candidates = []
    for p_idx, p in enumerate(pred_vals):
        for t_idx, t in enumerate(target_vals):
            denom = abs(t) if abs(t) > 1e-7 else 1e-7
            rne = abs(p - t) / denom
            if rne <= rel_tol:
                candidates.append((rne, abs(p - t), p_idx, t_idx))

    candidates.sort(key=lambda x: x[0])

    for rne, abs_err, p_idx, t_idx in candidates:
        if p_idx not in matched_preds and t_idx not in matched_targets:
            matched_preds.add(p_idx)
            matched_targets.add(t_idx)
            paired_errors.append(abs_err ** 2)
            paired_rnes.append(rne)

    tp = len(matched_preds)
    precision = tp / len(pred_vals) if pred_vals else 0.0
    recall = tp / len(target_vals) if target_vals else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    rmse = math.sqrt(np.mean(paired_errors)) if paired_errors else float("nan")
    mean_rne = float(np.mean(paired_rnes)) if paired_rnes else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse": rmse,
        "mean_rne": mean_rne,
    }


class TableNode:
    """Represents a node in a table tree structure for TEDS computation."""

    def __init__(self, label: str, text: str = ""):
        self.label = label
        self.text = str(text).strip()
        self.children: List["TableNode"] = []


def parse_markdown_to_tree(md_text: str) -> TableNode:
    """Parses a Markdown table string into a TableNode tree."""
    from .table_parser import parse_markdown_to_dataframe

    df, is_valid = parse_markdown_to_dataframe(md_text)
    root = TableNode("table")
    if not is_valid or df is None:
        return root

    # thead / header row
    tr_head = TableNode("tr")
    for col in df.columns:
        tr_head.children.append(TableNode("th", text=str(col)))
    root.children.append(tr_head)

    # tbody / data rows
    for _, row in df.iterrows():
        tr_row = TableNode("tr")
        for val in row:
            tr_row.children.append(TableNode("td", text=str(val)))
        root.children.append(tr_row)

    return root


def compute_tree_edit_distance(tree1: TableNode, tree2: TableNode) -> Tuple[float, int, int]:
    """Computes tree edit distance using the Zhang-Shasha algorithm.
    
    Returns:
        Tuple of (tree_edit_distance, num_nodes_tree1, num_nodes_tree2)
    """
    def postorder(node: TableNode, nodes: List[TableNode]):
        for c in node.children:
            postorder(c, nodes)
        nodes.append(node)

    nodes1: List[TableNode] = []
    nodes2: List[TableNode] = []
    postorder(tree1, nodes1)
    postorder(tree2, nodes2)

    n1, n2 = len(nodes1), len(nodes2)
    if n1 == 0 and n2 == 0:
        return 0.0, 0, 0
    if n1 == 0:
        return float(n2), 0, n2
    if n2 == 0:
        return float(n1), n1, 0

    def get_lld(nodes: List[TableNode]) -> Dict[int, int]:
        lld = {}
        for i, node in enumerate(nodes):
            curr = node
            while curr.children:
                curr = curr.children[0]
            lld[i] = nodes.index(curr)
        return lld

    lld1 = get_lld(nodes1)
    lld2 = get_lld(nodes2)

    def get_keyroots(nodes: List[TableNode], lld: Dict[int, int]) -> List[int]:
        kr = []
        for i in range(len(nodes)):
            is_kr = True
            for j in range(i + 1, len(nodes)):
                if lld[j] == lld[i]:
                    is_kr = False
                    break
            if is_kr:
                kr.append(i)
        return kr

    kr1 = get_keyroots(nodes1, lld1)
    kr2 = get_keyroots(nodes2, lld2)

    def cost(u: TableNode, v: TableNode) -> float:
        if u.label != v.label:
            return 1.0
        if u.text or v.text:
            m = max(len(u.text), len(v.text), 1)
            return Levenshtein.distance(u.text, v.text) / m
        return 0.0

    treedist: Dict[Tuple[int, int], float] = {}
    fd: Dict[Tuple[int, int], float] = {}

    for i in kr1:
        for j in kr2:
            fd.clear()
            for p in range(lld1[i] - 1, i + 1):
                fd[(p, lld2[j] - 1)] = (p - lld1[i] + 1) * 1.0
            for q in range(lld2[j] - 1, j + 1):
                fd[(lld1[i] - 1, q)] = (q - lld2[j] + 1) * 1.0
            fd[(lld1[i] - 1, lld2[j] - 1)] = 0.0

            for p in range(lld1[i], i + 1):
                for q in range(lld2[j], j + 1):
                    c_del = fd[(p - 1, q)] + 1.0
                    c_ins = fd[(p, q - 1)] + 1.0
                    if lld1[p] == lld1[i] and lld2[q] == lld2[j]:
                        c_match = fd[(p - 1, q - 1)] + cost(nodes1[p], nodes2[q])
                        treedist[(p, q)] = c_match
                    else:
                        c_match = fd[(lld1[p] - 1, lld2[q] - 1)] + treedist.get((p, q), 1.0)
                    fd[(p, q)] = min(c_del, c_ins, c_match)

    ted = fd[(n1 - 1, n2 - 1)]
    return ted, n1, n2


def compute_teds(pred_md: str, target_md: str) -> float:
    """Computes Tree Edit Distance-based Similarity (TEDS) between two Markdown tables.
    
    Returns a score in [0.0, 1.0], where 1.0 indicates structural and content identity.
    If the predicted text is not a valid table while target is valid, returns 0.0.
    """
    from .table_parser import parse_markdown_to_dataframe

    if not pred_md and not target_md:
        return 1.0
    if not pred_md or not target_md:
        return 0.0

    _, is_pred_valid = parse_markdown_to_dataframe(pred_md)
    _, is_target_valid = parse_markdown_to_dataframe(target_md)

    if not is_pred_valid and not is_target_valid:
        return 1.0 if pred_md.strip() == target_md.strip() else 0.0
    if not is_pred_valid or not is_target_valid:
        return 0.0

    tree_pred = parse_markdown_to_tree(pred_md)
    tree_target = parse_markdown_to_tree(target_md)

    ted, n_pred, n_target = compute_tree_edit_distance(tree_pred, tree_target)
    max_nodes = max(n_pred, n_target)
    if max_nodes <= 1:
        return 1.0 if (n_pred == n_target) else 0.0

    return max(0.0, 1.0 - (ted / max_nodes))


def compute_icdar_rms(
    pred_df: Optional[pd.DataFrame],
    target_df: Optional[pd.DataFrame],
    rel_tol: float = 0.05
) -> float:
    """Computes the Relative Matching Score (RMS) for numerical cell values.
    
    Formula: RMS = cell_recall * max(0.0, 1.0 - mean_rne)
    Returns:
        float score in [0.0, 1.0].
    """
    metrics = compute_numerical_cell_metrics(pred_df, target_df, rel_tol=rel_tol)
    recall = metrics.get("recall", 0.0)
    rne = metrics.get("mean_rne", 0.0)
    if math.isnan(rne):
        rne = 0.0
    return max(0.0, min(1.0, recall * max(0.0, 1.0 - rne)))


def compute_icdar_score(
    pred_md: str,
    target_md: str,
    rel_tol: float = 0.05
) -> Dict[str, float]:
    """Computes official ICDAR 2026 Task 2 composite metrics: TEDS, RMS, and composite score.
    
    Returns:
        Dict with keys 'teds', 'rms', and 'icdar_score' (average of TEDS and RMS, [0, 1]).
    """
    from .table_parser import parse_markdown_to_dataframe

    teds = compute_teds(pred_md, target_md)
    pred_df, _ = parse_markdown_to_dataframe(pred_md)
    target_df, _ = parse_markdown_to_dataframe(target_md)
    rms = compute_icdar_rms(pred_df, target_df, rel_tol=rel_tol)
    composite = 0.5 * (teds + rms)

    return {
        "teds": teds,
        "rms": rms,
        "icdar_score": composite,
    }
