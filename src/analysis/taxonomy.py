"""
Rule-based error categories for one panel, plus a flag for suspicious ground truth.

Thresholds are first guesses; they get calibrated against the manual review
(see src/analysis/calibration.py).
"""

import math
import re
import statistics
from typing import List, Optional, Tuple

from rapidfuzz import fuzz, utils

from .scoring import is_valid_table

CATEGORIES = (
    "painel_faltando",
    "tabela_invalida",
    "colunas_divergentes",
    "amostragem_linhas",
    "cabecalho_desalinhado",
    "escala_magnitude",
    "grade_regular_inventada",
    "numero_nao_numerico",
)

ROW_RATIO_RANGE = (0.67, 1.5)
HEADER_SIMILARITY_MIN = 60
MAGNITUDE_MIN_LOG10 = 0.9
MAGNITUDE_MAX_DIST_TO_INT = 0.15
REGULAR_STEP_TOLERANCE = 0.01
SUSPICIOUS_GT_MAX_ROWS = 50

APPROX_RE = re.compile(r"~|≈|approx|10\^", re.IGNORECASE)
PLACEHOLDERS = {"", "-", "--", "—", "n/a", "na"}
DECIMAL_COMMA_RE = re.compile(r"^-?\d+,(\d{1,2}|\d{4,})$")
RANGE_RE = re.compile(r"^\d+(\.\d+)?\s*[-–]\s*\d")


def parse_table(md: str) -> Tuple[List[str], List[List[str]]]:
    """Splits a markdown table into (headers, data rows); the separator line is skipped."""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    if not cells:
        return [], []
    body = cells[2:] if len(cells) > 1 and all(re.fullmatch(r":?-+:?", c) for c in cells[1]) else cells[1:]
    return cells[0], body


def to_number(cell: str) -> Optional[float]:
    try:
        return float(cell.rstrip("%"))
    except ValueError:
        return None


def column(rows: List[List[str]], idx: int) -> List[str]:
    return [r[idx] for r in rows if idx < len(r)]


def numbers(cells: List[str]) -> List[float]:
    return [n for n in map(to_number, cells) if n is not None]


def is_regular(values: List[float]) -> bool:
    """True when consecutive steps are all equal within REGULAR_STEP_TOLERANCE of the mean step."""
    steps = [b - a for a, b in zip(values, values[1:])]
    mean = statistics.fmean(steps)
    return mean != 0 and all(abs(s - mean) <= REGULAR_STEP_TOLERANCE * abs(mean) for s in steps)


def _magnitude_off(gt_rows, pred_rows, n_cols: int) -> bool:
    # First numeric column after the x column, compared row by row.
    for idx in range(1, n_cols):
        pairs = [
            (g, p) for g, p in zip(map(to_number, column(gt_rows, idx)), map(to_number, column(pred_rows, idx)))
            if g and p
        ]
        if len(pairs) >= 2:
            ratio = statistics.median(p / g for g, p in pairs)
            if ratio <= 0:
                return False
            mag = abs(math.log10(ratio))
            return mag >= MAGNITUDE_MIN_LOG10 and abs(mag - round(mag)) < MAGNITUDE_MAX_DIST_TO_INT
    return False


def classify_panel(gt_table: str, pred_table: Optional[str]) -> List[str]:
    """Returns every error category the panel's prediction falls into (empty if none)."""
    if not (pred_table or "").strip():
        return ["painel_faltando"]
    if not is_valid_table(pred_table):
        return ["tabela_invalida"]

    gt_headers, gt_rows = parse_table(gt_table)
    pred_headers, pred_rows = parse_table(pred_table)
    found = []

    if len(pred_headers) != len(gt_headers):
        found.append("colunas_divergentes")

    if gt_rows:
        ratio = len(pred_rows) / len(gt_rows)
        if not ROW_RATIO_RANGE[0] <= ratio <= ROW_RATIO_RANGE[1]:
            found.append("amostragem_linhas")

    if len(pred_headers) == len(gt_headers) and gt_headers:
        sims = [fuzz.token_sort_ratio(g, p, processor=utils.default_process) for g, p in zip(gt_headers, pred_headers)]
        if statistics.fmean(sims) < HEADER_SIMILARITY_MIN:
            found.append("cabecalho_desalinhado")

    if _magnitude_off(gt_rows, pred_rows, min(len(gt_headers), len(pred_headers))):
        found.append("escala_magnitude")

    pred_x, gt_x = numbers(column(pred_rows, 0)), numbers(column(gt_rows, 0))
    if len(pred_x) >= 3 and is_regular(pred_x) and not (len(gt_x) >= 3 and is_regular(gt_x)):
        found.append("grade_regular_inventada")

    if any(APPROX_RE.search(c) for row in pred_rows for c in row):
        found.append("numero_nao_numerico")

    return found


def flag_suspicious_gt(gt_table: str) -> bool:
    """Flags ground truth that no prediction can match cleanly (noise in the annotation)."""
    headers, rows = parse_table(gt_table)
    if len(rows) > SUSPICIOUS_GT_MAX_ROWS:
        return True

    cells = [c for row in rows for c in row]
    if any(";" in c or APPROX_RE.search(c) or DECIMAL_COMMA_RE.match(c) or RANGE_RE.match(c) for c in cells):
        return True

    for idx in range(len(headers)):
        values = [c for c in column(rows, idx) if c.lower() not in PLACEHOLDERS]
        n_numeric = len(numbers(values))
        if n_numeric * 2 > len(values) and n_numeric < len(values):
            return True
    return False
