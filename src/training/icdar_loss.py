"""
Loss functions aligned with official ICDAR 2026 Task 2 metrics (TEDS and RMS).
Supports:
1. Float-Proximity & Point-Cardinality Aware Token-Weighted Surrogate Loss (100% forward-pass compatible, low VRAM).
2. Self-Critical Sequence Training (SCST / Policy Gradient) with Float-Proximity & Point-Count Reward.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.table_metrics import compute_icdar_score, sanitize_numerical_cell
from ..metrics.table_parser import extract_all_markdown_table_blocks
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

_FLOAT_REGEX = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def compute_float_proximity_weight(
    pred_val: Optional[float],
    target_val: float,
    near_tol: float = 0.01,
    rel_tol: float = 0.05,
    min_near_weight: float = 0.05,
    max_numeric_weight: float = 4.0,
) -> Tuple[float, float]:
    """Computes a smooth loss weight based on relative numerical distance in float space.

    Behavior:
    - If relative error <= near_tol (e.g. 1%): almost zero penalty (`min_near_weight`, e.g. 0.05).
    - If near_tol < relative error <= rel_tol (e.g. 1% to 5%): smooth quadratic ramp from `min_near_weight` to 1.0.
    - If relative error > rel_tol (e.g. > 5%): smooth ramp from 1.0 up to `max_numeric_weight` (saturating at 25% error).
    - If pred_val is None (failed to produce a number where target has a float): `max_numeric_weight`.

    Returns:
        Tuple of (weight, rel_error)
    """
    if pred_val is None or math.isnan(pred_val) or math.isinf(pred_val):
        return float(max_numeric_weight), 1.0

    denom = max(abs(target_val), 1e-7)
    rel_err = abs(pred_val - target_val) / denom

    if rel_err <= near_tol:
        # Within 1%: barely penalize at all, scaling gently from min_near_weight * 0.5 to min_near_weight
        ratio = rel_err / max(near_tol, 1e-8)
        weight = min_near_weight * (0.5 + 0.5 * ratio)
        return float(weight), float(rel_err)

    if rel_err <= rel_tol:
        # Between 1% and 5%: smooth quadratic progression from min_near_weight to 1.0
        span = max(rel_tol - near_tol, 1e-8)
        t = (rel_err - near_tol) / span
        weight = min_near_weight + (1.0 - min_near_weight) * (t ** 2)
        return float(weight), float(rel_err)

    # Above rel_tol (e.g. > 5%): increase penalty toward max_numeric_weight
    far_span = max(0.25 - rel_tol, 1e-8)
    t_far = min(1.0, (rel_err - rel_tol) / far_span)
    weight = 1.0 + (max_numeric_weight - 1.0) * t_far
    return float(weight), float(rel_err)


def extract_floats_from_text(text: str) -> List[float]:
    """Extracts all float values from a decoded Markdown table string."""
    if not text:
        return []
    # Normalize European decimal commas when surrounded by digits (e.g. 21,78 -> 21.78)
    normalized = re.sub(r"(?<=\d),(?=\d{1,2}(?:\D|$))", ".", text)
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalized)
    vals: List[float] = []
    for m in _FLOAT_REGEX.finditer(normalized):
        s = m.group(0)
        try:
            v = float(s)
            if not (math.isnan(v) or math.isinf(v)):
                vals.append(v)
        except ValueError:
            continue
    return vals


def count_data_rows_and_tables(text: str) -> Tuple[int, int]:
    """Counts total data rows and total Markdown tables in a text."""
    rows, tables, _ = count_data_rows_tables_and_cols(text)
    return rows, tables


def count_data_rows_tables_and_cols(text: str) -> Tuple[int, int, int]:
    """Counts total data rows, total Markdown tables, and total columns across tables in a text."""
    if not text:
        return 0, 0, 0
    blocks = extract_all_markdown_table_blocks(text)
    if blocks:
        num_tables = len(blocks)
        data_rows = 0
        total_cols = 0
        for b in blocks:
            lines = [ln.strip() for ln in b.strip().split("\n") if ln.strip()]
            if lines:
                header_cells = [c.strip() for c in lines[0].split("|")[1:-1]]
                total_cols += max(1, len(header_cells))
            # Subtract header and separator line
            data_rows += max(0, len(lines) - 2)
        return data_rows, num_tables, total_cols

    # Fallback line-based count for partial/toy sequences in unit tests
    lines = [ln.strip() for ln in text.strip().split("\n") if "|" in ln]
    cols = max(1, len(lines[0].split("|")[1:-1])) if lines else 0
    return max(0, len(lines)), (1 if lines else 0), cols


def compute_cardinality_penalty(
    num_pred_points: int,
    num_target_points: int,
    num_pred_rows: int = 0,
    num_target_rows: int = 0,
    num_pred_tables: int = 1,
    num_target_tables: int = 1,
    num_pred_cols: int = 0,
    num_target_cols: int = 0,
    count_penalty_weight: float = 2.0,
    table_count_penalty_weight: float = 3.5,
    col_count_penalty_weight: float = 3.5,
) -> Tuple[float, Dict[str, float]]:
    """Computes a penalty multiplier when predicted point/row/table/column counts differ from ground truth.

    Returns:
        Tuple of (multiplier >= 1.0, stats_dict)
    """
    if num_target_points > 0:
        point_rel_diff = abs(num_pred_points - num_target_points) / float(num_target_points)
    else:
        point_rel_diff = 1.0 if num_pred_points > 0 else 0.0

    if num_target_rows > 0:
        row_rel_diff = abs(num_pred_rows - num_target_rows) / float(num_target_rows)
    else:
        row_rel_diff = 0.0

    if num_target_tables > 0:
        table_rel_diff = abs(num_pred_tables - num_target_tables) / float(num_target_tables)
    else:
        table_rel_diff = 0.0

    if num_target_cols > 0:
        col_rel_diff = abs(num_pred_cols - num_target_cols) / float(num_target_cols)
    else:
        col_rel_diff = 0.0

    combined_point_diff = max(point_rel_diff, row_rel_diff)
    penalty_points = count_penalty_weight * min(2.0, combined_point_diff)
    penalty_tables = table_count_penalty_weight * min(2.5, table_rel_diff)
    penalty_cols = col_count_penalty_weight * min(2.5, col_rel_diff)
    multiplier = 1.0 + penalty_points + penalty_tables + penalty_cols

    return float(multiplier), {
        "point_rel_diff": float(point_rel_diff),
        "row_rel_diff": float(row_rel_diff),
        "table_rel_diff": float(table_rel_diff),
        "col_rel_diff": float(col_rel_diff),
        "cardinality_multiplier": float(multiplier),
    }


class ICDARGPUTensorReward(nn.Module):
    """Reward module computing Float-Proximity Numerical Score, Point-Count Penalty, and Structural TEDS proxy.

    Upgraded from token-ID bincount to true float conversion and relative percentage tolerance:
    - Converts extracted numbers to `float` and rewards proximity (<= 1% gets ~1.0 reward, decaying smoothly).
    - Penalizes mismatches in the number of extracted points/rows and number of tables.
    """

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: Optional[int] = None,
        near_tol: float = 0.01,
        rel_tol: float = 0.05,
        count_penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.near_tol = near_tol
        self.rel_tol = rel_tol
        self.count_penalty_weight = count_penalty_weight
        tok_vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 0
        self.vocab_size = max(tok_vocab_size, vocab_size or 0, 151936)

        is_numeric = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_structural = torch.zeros(self.vocab_size, dtype=torch.bool)

        vocab_dict = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        for token, idx in vocab_dict.items():
            if idx >= self.vocab_size:
                continue
            clean_tok = token.replace("Ġ", "").replace(" ", "").strip()
            if clean_tok and all(c in "0123456789.-" for c in clean_tok):
                is_numeric[idx] = True
            elif clean_tok in ("|", "-", "--", "---", "\n"):
                is_structural[idx] = True

        self.register_buffer("is_numeric", is_numeric)
        self.register_buffer("is_structural", is_structural)

    def forward(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes composite float-aware and cardinality-aware reward."""
        pred_tokens = pred_tokens.view(-1)
        target_tokens = target_tokens.view(-1)
        device = pred_tokens.device

        # Decode token sequences to extract actual floats and table counts
        pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else ""
        target_text = self.tokenizer.decode(target_tokens, skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else ""

        pred_floats = extract_floats_from_text(pred_text)
        target_floats = extract_floats_from_text(target_text)

        # 1. Float-proximity numerical reward + point-count penalty
        if not target_floats:
            r_num_val = 1.0 if not pred_floats else 0.0
            count_factor = 1.0
        elif not pred_floats:
            r_num_val = 0.0
            count_factor = 0.0
        else:
            # Greedy minimum-relative-error bipartite matching in float space
            candidates = []
            for p_idx, p_val in enumerate(pred_floats):
                for t_idx, t_val in enumerate(target_floats):
                    denom = max(abs(t_val), 1e-7)
                    rel_err = abs(p_val - t_val) / denom
                    candidates.append((rel_err, p_idx, t_idx))
            candidates.sort(key=lambda x: x[0])

            matched_p = set()
            matched_t = set()
            pair_scores = []
            for rel_err, p_idx, t_idx in candidates:
                if p_idx not in matched_p and t_idx not in matched_t:
                    matched_p.add(p_idx)
                    matched_t.add(t_idx)
                    if rel_err <= self.near_tol:
                        # Within 1%: virtually no penalty (score ~ 1.0)
                        score = 1.0 - 0.02 * (rel_err / max(self.near_tol, 1e-8))
                    elif rel_err <= self.rel_tol:
                        # 1% to 5%: smooth decay from 0.98 down to 0.70
                        t = (rel_err - self.near_tol) / max(self.rel_tol - self.near_tol, 1e-8)
                        score = 0.98 - 0.28 * (t ** 2)
                    elif rel_err <= 0.20:
                        # 5% to 20%: decays from 0.70 down to 0.0
                        t = (rel_err - self.rel_tol) / (0.20 - self.rel_tol)
                        score = max(0.0, 0.70 * (1.0 - t))
                    else:
                        score = 0.0
                    pair_scores.append(score)

            sum_scores = sum(pair_scores)
            precision_float = sum_scores / len(pred_floats)
            recall_float = sum_scores / len(target_floats)
            f1_float = (
                (2.0 * precision_float * recall_float) / (precision_float + recall_float + 1e-8)
                if (precision_float + recall_float) > 0
                else 0.0
            )

            # Explicit point-count penalty factor
            point_diff_ratio = abs(len(pred_floats) - len(target_floats)) / float(max(len(target_floats), 1))
            count_factor = max(0.0, 1.0 - self.count_penalty_weight * 0.5 * point_diff_ratio)
            r_num_val = f1_float * count_factor

        r_num = torch.tensor(r_num_val, device=device, dtype=torch.float32)

        # 2. Structural fidelity proxy + multi-table count penalty
        struct_mask_p = self.is_structural[pred_tokens.clamp(0, self.vocab_size - 1)]
        struct_mask_t = self.is_structural[target_tokens.clamp(0, self.vocab_size - 1)]
        pred_struct = pred_tokens[struct_mask_p]
        target_struct = target_tokens[struct_mask_t]

        if target_struct.numel() == 0:
            r_struct = torch.tensor(1.0 if pred_struct.numel() == 0 else 0.0, device=device)
        elif pred_struct.numel() == 0:
            r_struct = torch.tensor(0.0, device=device)
        else:
            c_p = torch.bincount(pred_struct, minlength=self.vocab_size)
            c_t = torch.bincount(target_struct, minlength=self.vocab_size)
            inter_1 = torch.minimum(c_p, c_t).sum().float()
            dice_1 = (2.0 * inter_1) / (pred_struct.numel() + target_struct.numel() + 1e-8)

            if pred_struct.numel() > 1 and target_struct.numel() > 1:
                bi_p = pred_struct[:-1] * self.vocab_size + pred_struct[1:]
                bi_t = target_struct[:-1] * self.vocab_size + target_struct[1:]
                inter_2 = torch.isin(bi_p, bi_t).sum().float()
                dice_2 = (2.0 * inter_2) / (bi_p.numel() + bi_t.numel() + 1e-8)
                r_struct = 0.5 * (dice_1 + dice_2)
            else:
                r_struct = dice_1

            # Penalize structural reward if number of tables, columns, or rows diverges
            pred_rows, pred_tables, pred_cols = count_data_rows_tables_and_cols(pred_text)
            targ_rows, targ_tables, targ_cols = count_data_rows_tables_and_cols(target_text)
            if targ_tables > 0 and pred_tables != targ_tables:
                table_pen = max(0.2, 1.0 - 0.5 * (abs(pred_tables - targ_tables) / float(targ_tables)))
                r_struct = r_struct * table_pen
            if targ_cols > 0 and pred_cols != targ_cols:
                col_pen = max(0.2, 1.0 - 0.5 * (abs(pred_cols - targ_cols) / float(targ_cols)))
                r_struct = r_struct * col_pen
            if targ_rows > 0 and pred_rows != targ_rows:
                row_pen = max(0.3, 1.0 - 0.35 * (abs(pred_rows - targ_rows) / float(targ_rows)))
                r_struct = r_struct * row_pen

        total_reward = 0.5 * (r_num + r_struct)
        return total_reward, {
            "r_num": r_num.item(),
            "r_struct": r_struct.item(),
            "count_factor": float(count_factor),
            "reward": total_reward.item(),
        }


class ICDARTokenWeightedLoss(nn.Module):
    """Differentiable surrogate loss with:
    1. Float-proximity dynamic token weighting (converts predicted vs target numeric spans to float:
       <= 1% error is barely penalized, ramping smoothly up to 5% and beyond).
    2. Differentiable Soft-Digit Expected Value Loss on digit tokens (0-9).
    3. Explicit Cardinality Penalty when the number of extracted points/rows/tables/columns differs from target.
    4. Header & Separator Structure Boosting for TEDS topology fidelity and directional EOS/Newline control.
    """

    def __init__(
        self,
        tokenizer: Any,
        numeric_weight: float = 4.0,
        structural_weight: float = 4.5,
        header_weight: float = 6.0,
        text_weight: float = 2.5,
        vocab_size: Optional[int] = None,
        focal_gamma: float = 0.0,
        near_tol: float = 0.01,
        rel_tol: float = 0.05,
        min_near_weight: float = 0.20,
        count_penalty_weight: float = 2.0,
        table_count_penalty_weight: float = 3.5,
        col_count_penalty_weight: float = 3.5,
        soft_digit_weight: float = 0.25,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.numeric_weight = float(numeric_weight)
        self.structural_weight = float(structural_weight)
        self.header_weight = float(header_weight)
        self.text_weight = float(text_weight)
        self.focal_gamma = float(focal_gamma)
        self.near_tol = float(near_tol)
        self.rel_tol = float(rel_tol)
        self.min_near_weight = float(min_near_weight)
        self.count_penalty_weight = float(count_penalty_weight)
        self.table_count_penalty_weight = float(table_count_penalty_weight)
        self.col_count_penalty_weight = float(col_count_penalty_weight)
        self.soft_digit_weight = float(soft_digit_weight)

        tok_vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 0
        self.vocab_size = max(tok_vocab_size, vocab_size or 0, 151936)
        self.last_stats: Dict[str, float] = {}
        self._build_token_weights()

    def _build_token_weights(self):
        """Precomputes token masks, static fallback weights, and digit token mappings."""
        weights = torch.ones(self.vocab_size, dtype=torch.float32)
        is_numeric_tok = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_structural_tok = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_text_tok = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_newline_tok = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_eos_tok = torch.zeros(self.vocab_size, dtype=torch.bool)
        is_newline_or_eos = torch.zeros(self.vocab_size, dtype=torch.bool)
        digit_value_map = torch.full((self.vocab_size,), -1.0, dtype=torch.float32)

        # Single-digit token IDs for differentiable expected-digit loss
        digit_token_ids: List[int] = []
        digit_token_vals: List[float] = []

        vocab_dict = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        for token, idx in vocab_dict.items():
            if idx >= self.vocab_size:
                continue
            decoded = self.tokenizer.decode([idx]) if hasattr(self.tokenizer, "decode") else token
            clean = decoded.strip()
            if not clean:
                if "\n" in decoded:
                    weights[idx] = self.structural_weight
                    is_structural_tok[idx] = True
                    is_newline_tok[idx] = True
                    is_newline_or_eos[idx] = True
                continue
            if "|" in decoded or all(c in "-:|\n\r " for c in decoded) or decoded in ("\n", "\n\n"):
                weights[idx] = self.structural_weight
                is_structural_tok[idx] = True
                if "\n" in decoded:
                    is_newline_tok[idx] = True
                    is_newline_or_eos[idx] = True
            elif all(c in "0123456789.-+eE" for c in clean) and (
                any(c.isdigit() for c in clean) or clean in (".", "-", "+")
            ):
                weights[idx] = self.numeric_weight
                is_numeric_tok[idx] = True
                if len(clean) == 1 and clean.isdigit():
                    d_val = float(int(clean))
                    digit_value_map[idx] = d_val
                    if d_val not in digit_token_vals:
                        digit_token_ids.append(idx)
                        digit_token_vals.append(d_val)
            elif any(c.isalpha() for c in clean):
                weights[idx] = self.text_weight
                is_text_tok[idx] = True

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and 0 <= eos_id < self.vocab_size:
            is_eos_tok[eos_id] = True
            is_newline_or_eos[eos_id] = True
            weights[eos_id] = max(weights[eos_id].item(), self.structural_weight)

        self.register_buffer("token_weights", weights)
        self.register_buffer("is_numeric_tok", is_numeric_tok)
        self.register_buffer("is_structural_tok", is_structural_tok)
        self.register_buffer("is_text_tok", is_text_tok)
        self.register_buffer("is_newline_tok", is_newline_tok)
        self.register_buffer("is_eos_tok", is_eos_tok)
        self.register_buffer("is_newline_or_eos", is_newline_or_eos)
        self.register_buffer("digit_value_map", digit_value_map)
        if digit_token_ids:
            order = sorted(range(len(digit_token_ids)), key=lambda i: digit_token_vals[i])
            sorted_ids = [digit_token_ids[i] for i in order]
            sorted_vals = [digit_token_vals[i] for i in order]
            self.register_buffer("digit_ids", torch.tensor(sorted_ids, dtype=torch.long))
            self.register_buffer("digit_vals", torch.tensor(sorted_vals, dtype=torch.float32))
        else:
            self.digit_ids = None
            self.digit_vals = None

    @property
    def weights(self) -> torch.Tensor:
        return self.token_weights

    def compute_dynamic_float_and_cardinality_weights(
        self,
        pred_token_ids: torch.Tensor,
        valid_labels: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """Computes per-token float-proximity weights, TEDS header/column weights, and cardinality multiplier.

        1. Identifies contiguous numeric token spans in `valid_labels`.
        2. Decodes both target span and corresponding predicted tokens (`pred_token_ids`) into floats.
        3. Applies `compute_float_proximity_weight`:
           - <= 1% error -> `min_near_weight` (~0.12, low penalty while preserving string format)
           - 1% to 5% error -> smooth quadratic ramp up to 1.0
           - > 5% error -> ramps up to `numeric_weight` (4.0)
        4. Boosts tokens in the first 2 lines of each Markdown table (header `<th>` + separator `|---|`)
           to enforce correct column count and header names for TEDS tree structure.
        5. Compares total extracted float count, row count, table count, and column count between predicted
           and target sequence, applying directional EOS (anti-overgeneration) or Newline (anti-early-stop) weighting.
        """
        device = valid_labels.device
        dynamic_weights = base_weights.clone()

        lbl_clamped = valid_labels.clamp(0, self.vocab_size - 1)
        lbl_list = valid_labels.detach().cpu().tolist()
        pred_list = pred_token_ids.detach().cpu().tolist()
        num_mask = self.is_numeric_tok[lbl_clamped].detach().cpu().tolist()
        text_mask_cpu = self.is_text_tok[lbl_clamped].detach().cpu().tolist()
        nl_mask_cpu = self.is_newline_tok[lbl_clamped].detach().cpu().tolist()
        struct_mask_cpu = self.is_structural_tok[lbl_clamped].detach().cpu().tolist()

        # Boost tokens in the first 2 lines of each table (Header row + Separator row) for TEDS topology
        line_idx_in_table = 0
        consecutive_nl = 0
        header_text_weight = max(self.text_weight, self.header_weight * 0.9)
        for idx_t in range(len(lbl_list)):
            if nl_mask_cpu[idx_t]:
                tok_str = self.tokenizer.decode([lbl_list[idx_t]], skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else "\n"
                nl_count = max(1, tok_str.count("\n"))
                consecutive_nl += nl_count
                if line_idx_in_table < 2:
                    dynamic_weights[idx_t] = max(dynamic_weights[idx_t].item(), self.header_weight)
                if consecutive_nl >= 2:
                    # Double newline starts a new table block
                    line_idx_in_table = 0
                else:
                    line_idx_in_table += nl_count
            else:
                consecutive_nl = 0
                if line_idx_in_table < 2:
                    target_w = self.header_weight if struct_mask_cpu[idx_t] else header_text_weight
                    if dynamic_weights[idx_t].item() < target_w:
                        dynamic_weights[idx_t] = target_w
                else:
                    # In data rows: ensure textual tokens (units, category labels, chemical descriptors) receive text_weight
                    if text_mask_cpu[idx_t]:
                        if dynamic_weights[idx_t].item() < self.text_weight:
                            dynamic_weights[idx_t] = self.text_weight

        n_tokens = len(lbl_list)
        rel_errors: List[float] = []
        i = 0
        while i < n_tokens:
            if not num_mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n_tokens and num_mask[j]:
                j += 1

            # Contiguous numeric token span [i:j]
            targ_span_str = self.tokenizer.decode(lbl_list[i:j], skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else ""
            pred_span_str = self.tokenizer.decode(pred_list[i:j], skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else ""

            targ_float = sanitize_numerical_cell(targ_span_str)
            pred_float = sanitize_numerical_cell(pred_span_str)

            if targ_float is not None:
                w_span, rel_err = compute_float_proximity_weight(
                    pred_val=pred_float,
                    target_val=targ_float,
                    near_tol=self.near_tol,
                    rel_tol=self.rel_tol,
                    min_near_weight=self.min_near_weight,
                    max_numeric_weight=self.numeric_weight,
                )
                dynamic_weights[i:j] = w_span
                rel_errors.append(rel_err)

            i = j

        # Decode full sequence to check global float count, row count, table count, and column count
        full_targ_str = self.tokenizer.decode(lbl_list, skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else ""
        full_pred_str = self.tokenizer.decode(pred_list, skip_special_tokens=True) if hasattr(self.tokenizer, "decode") else ""

        targ_floats = extract_floats_from_text(full_targ_str)
        pred_floats = extract_floats_from_text(full_pred_str)
        targ_rows, targ_tables, targ_cols = count_data_rows_tables_and_cols(full_targ_str)
        pred_rows, pred_tables, pred_cols = count_data_rows_tables_and_cols(full_pred_str)

        card_mult, card_stats = compute_cardinality_penalty(
            num_pred_points=len(pred_floats),
            num_target_points=len(targ_floats),
            num_pred_rows=pred_rows,
            num_target_rows=targ_rows,
            num_pred_tables=pred_tables,
            num_target_tables=targ_tables,
            num_pred_cols=pred_cols,
            num_target_cols=targ_cols,
            count_penalty_weight=self.count_penalty_weight,
            table_count_penalty_weight=self.table_count_penalty_weight,
            col_count_penalty_weight=self.col_count_penalty_weight,
        )

        # Directional EOS vs Newline control to prevent both early stopping and row over-generation
        if card_mult > 1.05:
            if pred_rows > targ_rows or len(pred_floats) > len(targ_floats):
                # Over-generation: strongly boost EOS token so model learns to stop cleanly at table end
                eos_mask = self.is_eos_tok[lbl_clamped].to(device)
                dynamic_weights = torch.where(eos_mask, dynamic_weights * (card_mult * 1.5), dynamic_weights)
            if pred_rows < targ_rows or pred_tables < targ_tables:
                # Under-generation / early stopping: strongly boost newline and structural tokens to continue rows/tables
                nl_mask = self.is_newline_tok[lbl_clamped].to(device)
                dynamic_weights = torch.where(nl_mask, dynamic_weights * (card_mult * 1.5), dynamic_weights)
                struct_mask = self.is_structural_tok[lbl_clamped].to(device)
                dynamic_weights = torch.where(struct_mask, dynamic_weights * 1.3, dynamic_weights)
            if pred_cols != targ_cols:
                # Column count mismatch: reinforce all structural pipe/separator tokens and headers
                struct_mask = self.is_structural_tok[lbl_clamped].to(device)
                col_boost = 1.0 + min(2.0, card_stats.get("col_rel_diff", 0.0) * 1.5)
                dynamic_weights = torch.where(struct_mask, dynamic_weights * col_boost, dynamic_weights)

        stats = {
            "mean_float_rel_error": float(sum(rel_errors) / len(rel_errors)) if rel_errors else 0.0,
            "num_target_points": float(len(targ_floats)),
            "num_pred_points": float(len(pred_floats)),
            "num_target_cols": float(targ_cols),
            "num_pred_cols": float(pred_cols),
            **card_stats,
        }
        self.last_stats = stats
        return dynamic_weights, card_mult, stats

    def compute_soft_digit_loss(
        self,
        logits_2d: torch.Tensor,
        labels_1d: torch.Tensor,
        token_weights_1d: torch.Tensor,
    ) -> torch.Tensor:
        """Computes a differentiable expected-digit numerical distance on digit tokens (0-9).

        Allows gradient flow that distinguishes nearby digits (e.g. '9' vs '8') from far digits ('9' vs '0').
        """
        if self.digit_ids is None or self.digit_vals is None or self.soft_digit_weight <= 0.0:
            return torch.tensor(0.0, device=logits_2d.device)

        vocab_limit = logits_2d.shape[-1]
        valid_digit_mask = self.digit_ids < vocab_limit
        if not valid_digit_mask.all():
            return torch.tensor(0.0, device=logits_2d.device)

        lbl_clamped = labels_1d.clamp(0, self.vocab_size - 1)
        target_digit_vals = self.digit_value_map.to(labels_1d.device)[lbl_clamped]
        is_digit_pos = target_digit_vals >= 0.0
        if not is_digit_pos.any():
            return torch.tensor(0.0, device=logits_2d.device)

        digit_logits = logits_2d[is_digit_pos][:, self.digit_ids.to(logits_2d.device)]
        digit_probs = F.softmax(digit_logits.float(), dim=-1)
        expected_digit = torch.sum(digit_probs * self.digit_vals.to(logits_2d.device).unsqueeze(0), dim=-1)

        true_digit = target_digit_vals[is_digit_pos]
        # Normalized smooth L1 distance in [0, 1] over [0..9]
        digit_diff = torch.abs(expected_digit - true_digit) / 9.0
        # Modulate by the float-proximity weight so tokens already within 1% float error aren't over-penalized
        pos_weights = token_weights_1d[is_digit_pos]
        return (digit_diff * pos_weights).mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes float-proximity and cardinality-aware weighted cross-entropy loss over target tokens."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        flat_logits = shift_logits.view(-1, shift_logits.shape[-1])[valid_mask.view(-1)]
        flat_labels = shift_labels.view(-1)[valid_mask.view(-1)]

        w = self.token_weights.to(flat_logits.device)
        if w.shape[0] < flat_logits.shape[-1]:
            w = F.pad(w, (0, flat_logits.shape[-1] - w.shape[0]), value=1.0)
        elif w.shape[0] > flat_logits.shape[-1]:
            w = w[: flat_logits.shape[-1]]

        base_sample_weights = w[flat_labels.clamp(0, w.shape[0] - 1)]
        with torch.no_grad():
            pred_token_ids = flat_logits.argmax(dim=-1)
            dynamic_weights, card_mult, _ = self.compute_dynamic_float_and_cardinality_weights(
                pred_token_ids=pred_token_ids,
                valid_labels=flat_labels,
                base_weights=base_sample_weights,
            )

        ce = F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
        weighted_ce = ce * dynamic_weights

        if self.focal_gamma > 0.0:
            pt = torch.exp(-ce.detach())
            focal_w = (1.0 - pt) ** self.focal_gamma
            weighted_ce = focal_w * weighted_ce

        soft_digit = self.compute_soft_digit_loss(flat_logits, flat_labels, dynamic_weights)
        total_loss = (weighted_ce.mean() + self.soft_digit_weight * soft_digit) * card_mult
        return total_loss


class ICDARSCSTLoss(nn.Module):
    """Self-Critical Sequence Training (SCST) loss optimizing float-proximity RMS, point count, and TEDS."""

    def __init__(
        self,
        processor: Any,
        rel_tol: float = 0.05,
        near_tol: float = 0.01,
        temperature: float = 0.7,
        max_gen_tokens: int = 512,
        reward_mode: str = "gpu_tensor",
        vocab_size: Optional[int] = None,
        count_penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.processor = processor
        self.rel_tol = rel_tol
        self.near_tol = near_tol
        self.temperature = temperature
        self.max_gen_tokens = max_gen_tokens
        self.reward_mode = reward_mode
        self.gpu_reward = ICDARGPUTensorReward(
            processor.tokenizer,
            vocab_size=vocab_size,
            near_tol=near_tol,
            rel_tol=rel_tol,
            count_penalty_weight=count_penalty_weight,
        )

        # Pre-cache assistant header tokens for GPU sliding-window search
        im_start_fn = getattr(self.processor.tokenizer, "convert_tokens_to_ids", None)
        im_start_id = im_start_fn("<|im_start|>") if im_start_fn else None
        encode_fn = getattr(self.processor.tokenizer, "encode", None)
        assistant_ids = encode_fn("assistant", add_special_tokens=False) if encode_fn else []
        if im_start_id is not None and assistant_ids:
            h = [im_start_id] + assistant_ids + [198]  # with newline '\n'
            self.register_buffer("header_tensor", torch.tensor(h, dtype=torch.long))
        else:
            self.header_tensor = None

    def _find_prompt_end_idx(self, input_ids_row: torch.Tensor, labels_row: Optional[torch.Tensor] = None) -> int:
        """Finds prompt end index without CPU roundtrip whenever possible."""
        if labels_row is not None:
            mask = (labels_row != -100).nonzero(as_tuple=True)[0]
            if mask.numel() > 0 and mask[0].item() > 0:
                return mask[0].item()

        if self.header_tensor is not None and input_ids_row.numel() >= self.header_tensor.numel():
            h_tensor = self.header_tensor.to(input_ids_row.device)
            windows = input_ids_row.unfold(0, h_tensor.numel(), 1)
            matches = (windows == h_tensor).all(dim=-1).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                return (matches[0] + h_tensor.numel()).item()

        return input_ids_row.numel() // 2

    def compute_scst_sample_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        sample_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes SCST policy gradient loss for a single sample in the batch."""
        input_ids = inputs["input_ids"][sample_idx : sample_idx + 1]
        attention_mask = inputs["attention_mask"][sample_idx : sample_idx + 1]
        labels = inputs["labels"][sample_idx : sample_idx + 1]

        prompt_end_idx = self._find_prompt_end_idx(input_ids[0], labels[0])
        if prompt_end_idx <= 0 or prompt_end_idx >= input_ids.shape[1]:
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return zero_loss, {"r_sampled": 0.0, "r_greedy": 0.0, "advantage": 0.0}

        prompt_input_ids = input_ids[:, :prompt_end_idx]
        prompt_attention_mask = attention_mask[:, :prompt_end_idx]
        prompt_len = prompt_input_ids.shape[1]

        extra_kwargs = {}
        if "pixel_values" in inputs and "image_grid_thw" in inputs:
            grid_thw = inputs["image_grid_thw"]
            patches_per_image = [int(g[0].item() * g[1].item() * g[2].item()) for g in grid_thw]
            start_patch = sum(patches_per_image[:sample_idx])
            num_patches = patches_per_image[sample_idx]
            extra_kwargs["pixel_values"] = inputs["pixel_values"][start_patch : start_patch + num_patches]
            extra_kwargs["image_grid_thw"] = grid_thw[sample_idx : sample_idx + 1]

        for k, v in inputs.items():
            if k in ("labels", "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"):
                continue
            if isinstance(v, torch.Tensor) and v.shape[0] == inputs["input_ids"].shape[0]:
                extra_kwargs[k] = v[sample_idx : sample_idx + 1]
            else:
                extra_kwargs[k] = v

        target_tokens = input_ids[0, prompt_end_idx:]
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        eos_id = self.processor.tokenizer.eos_token_id

        with torch.no_grad():
            greedy_out = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **extra_kwargs,
            )
            sampled_out = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_gen_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                **extra_kwargs,
            )
            gen_tokens = sampled_out[0, prompt_len:]
            greedy_tokens = greedy_out[0, prompt_len:]

            if self.reward_mode == "gpu_tensor":
                r_greedy_t, _ = self.gpu_reward(greedy_tokens, target_tokens)
                r_sampled_t, _ = self.gpu_reward(gen_tokens, target_tokens)
                advantage = r_sampled_t - r_greedy_t
                reward_greedy = r_greedy_t.item()
                reward_sampled = r_sampled_t.item()
            else:
                pred_greedy = self.processor.decode(greedy_tokens, skip_special_tokens=True)
                pred_sampled = self.processor.decode(gen_tokens, skip_special_tokens=True)
                target_str = self.processor.decode(target_tokens, skip_special_tokens=True)
                reward_greedy = compute_icdar_score(pred_greedy, target_str, rel_tol=self.rel_tol)["icdar_score"]
                reward_sampled = compute_icdar_score(pred_sampled, target_str, rel_tol=self.rel_tol)["icdar_score"]
                advantage = torch.tensor(reward_sampled - reward_greedy, device=input_ids.device)

        info = {
            "r_sampled": reward_sampled,
            "r_greedy": reward_greedy,
            "advantage": advantage.item() if isinstance(advantage, torch.Tensor) else float(advantage),
        }

        if len(gen_tokens) == 0:
            zero_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return zero_loss, info

        forward_kwargs = dict(extra_kwargs)
        forward_kwargs["input_ids"] = sampled_out
        forward_kwargs["attention_mask"] = torch.ones_like(sampled_out)

        out = model(**forward_kwargs)
        logits = out.logits[0, prompt_len - 1 : -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=gen_tokens.unsqueeze(-1)).squeeze(-1)
        seq_log_prob = token_log_probs.mean()

        loss_scst = -advantage * seq_log_prob
        return loss_scst, info

    def forward(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes SCST loss averaged over batch."""
        batch_size = inputs["input_ids"].shape[0]
        losses = []
        sampled_rewards = []
        greedy_rewards = []
        advantages = []

        for b in range(batch_size):
            loss_b, info_b = self.compute_scst_sample_loss(model, inputs, sample_idx=b)
            losses.append(loss_b)
            sampled_rewards.append(info_b["r_sampled"])
            greedy_rewards.append(info_b["r_greedy"])
            advantages.append(info_b["advantage"])

        mean_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)
        stats = {
            "mean_r_sampled": sum(sampled_rewards) / len(sampled_rewards) if sampled_rewards else 0.0,
            "mean_r_greedy": sum(greedy_rewards) / len(greedy_rewards) if greedy_rewards else 0.0,
            "mean_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
        }
        return mean_loss, stats


class ICDARMetricLoss(nn.Module):
    """Hybrid loss module combining Float-Aware Token-Weighted SFT with optional SCST Metric Loss."""

    def __init__(
        self,
        processor: Any,
        loss_type: str = "scst",  # 'scst', 'token_weighted', 'hybrid'
        lambda_metric: float = 0.3,
        rel_tol: float = 0.05,
        near_tol: float = 0.01,
        min_near_weight: float = 0.20,
        temperature: float = 0.7,
        max_gen_tokens: int = 512,
        numeric_weight: float = 4.0,
        structural_weight: float = 4.5,
        header_weight: float = 6.0,
        text_weight: float = 2.5,
        count_penalty_weight: float = 2.0,
        table_count_penalty_weight: float = 3.5,
        col_count_penalty_weight: float = 3.5,
        soft_digit_weight: float = 0.25,
        reward_mode: str = "gpu_tensor",
        focal_gamma: float = 0.0,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_metric = lambda_metric
        self.processor = processor

        if loss_type in ("scst", "hybrid"):
            self.scst_module = ICDARSCSTLoss(
                processor=processor,
                rel_tol=rel_tol,
                near_tol=near_tol,
                temperature=temperature,
                max_gen_tokens=max_gen_tokens,
                reward_mode=reward_mode,
                vocab_size=vocab_size,
                count_penalty_weight=count_penalty_weight,
            )
        else:
            self.scst_module = None

        if loss_type in ("token_weighted", "hybrid"):
            self.weighted_module = ICDARTokenWeightedLoss(
                tokenizer=processor.tokenizer,
                numeric_weight=numeric_weight,
                structural_weight=structural_weight,
                header_weight=header_weight,
                text_weight=text_weight,
                vocab_size=vocab_size,
                focal_gamma=focal_gamma,
                near_tol=near_tol,
                rel_tol=rel_tol,
                min_near_weight=min_near_weight,
                count_penalty_weight=count_penalty_weight,
                table_count_penalty_weight=table_count_penalty_weight,
                col_count_penalty_weight=col_count_penalty_weight,
                soft_digit_weight=soft_digit_weight,
            )
        else:
            self.weighted_module = None

    def forward(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        outputs: Optional[Any] = None,
        sft_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the final combined loss and logs float-proximity and cardinality stats."""
        metrics_info: Dict[str, float] = {}

        if self.loss_type == "token_weighted":
            total_loss = outputs.loss if outputs is not None and outputs.loss is not None else sft_loss
            if total_loss is None:
                out = model(**inputs)
                total_loss = out.loss
            metrics_info["token_weighted_loss"] = total_loss.item()
            if self.weighted_module is not None and self.weighted_module.last_stats:
                metrics_info.update(self.weighted_module.last_stats)
            return total_loss, metrics_info

        # SCST or Hybrid
        if sft_loss is None:
            out = model(**inputs)
            sft_loss = out.loss

        metrics_info["sft_ce_loss"] = sft_loss.item()
        if self.weighted_module is not None and self.weighted_module.last_stats:
            metrics_info.update(self.weighted_module.last_stats)

        if self.lambda_metric <= 0.0 or self.scst_module is None:
            return sft_loss, metrics_info

        scst_loss, scst_stats = self.scst_module(model, inputs)
        metrics_info.update(scst_stats)
        metrics_info["scst_loss"] = scst_loss.item()

        total_loss = (1.0 - self.lambda_metric) * sft_loss + self.lambda_metric * scst_loss
        metrics_info["total_loss"] = total_loss.item()

        return total_loss, metrics_info
