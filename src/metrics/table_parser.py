"""
Robust Markdown table parsing and DataFrame extraction.
"""

import io
import re
from typing import Optional, Tuple, List
import pandas as pd


SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def sanitize_cell_text(text: str) -> str:
    """Sanitizes raw table cell text:
    - Strips approximation symbols (~, ≈, approx, ca., est.)
    - Normalizes unicode superscripts (e.g. 10⁶ -> 10^6)
    - Expands powers of 10 (e.g. ~10^7 -> 10000000.0)
    - Strips uncertainty expressions (87 ± 1.5 -> 87)
    - Cleans percentages and trailing physical units
    - Normalizes thousand commas (118,369.40 -> 118369.40)
    """
    if not text or not isinstance(text, str):
        return "" if text is None else str(text)

    s = text.strip()
    if not s or s in ("-", "--", "---", "N/A", "NA", "none", "None", "."):
        return s

    # Strip leading approximation prefixes
    s = re.sub(r"^(?:~|≈|approx\.?|ca\.?|est\.?)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[\s~≈]+$", "", s)

    # Convert unicode superscripts to standard numbers with ^
    if any(c in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺" for c in s):
        parts = re.split(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", s)
        res = []
        for p in parts:
            if any(c in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺" for c in p):
                res.append("^" + p.translate(SUPERSCRIPT_MAP))
            else:
                res.append(p)
        s = "".join(res)

    # Convert scientific power notation: (coeff x)? 10^[+-]?exp
    pow_match = re.search(r"^(?:(\d+(?:\.\d+)?)\s*[xX*×]\s*)?10\^([+-]?\d+)$", s)
    if pow_match:
        coeff = float(pow_match.group(1)) if pow_match.group(1) else 1.0
        exp = int(pow_match.group(2))
        try:
            val = coeff * (10.0 ** exp)
            return f"{val:.8g}"
        except OverflowError:
            pass

    # Strip percentage
    s = re.sub(r"%\s*$", "", s)

    # Strip uncertainty notations: ± 1.5, +/- 1.5, (5)
    s = re.sub(r"\s*(?:±|\+/-|\+-)\s*[\d.]+", "", s)
    s = re.sub(r"\(\d+\)$", "", s)

    # Strip trailing common units when preceded by a number
    s = re.sub(
        r"(?<=\d)\s*(?:s|ms|ns|nm|um|µm|mm|cm|m|km|g|mg|kg|deg|°C|°|K|counts|a\.u\.|eV|keV|MeV|Hz|kHz|MHz|GHz|Pa|kPa|MPa|Torr|V|mV|A|mA)\.?$",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Handle thousands comma vs European decimal comma
    if re.search(r"\d+,\d{3}", s):
        s = s.replace(",", "")
    elif s.count(",") == 1 and "." not in s and re.search(r"^\d+,\d+$", s):
        s = s.replace(",", ".")

    return s.strip()


def extract_all_markdown_table_blocks(text: str) -> List[str]:
    """Extracts all markdown table sections from generated or ground-truth text.
    Handles multiple tables (e.g. multi-panel figures separated by text or blank lines).
    """
    if not text or not isinstance(text, str):
        return []

    # Strip code fences if present
    code_block_match = re.search(r"```(?:markdown|table)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate_text = code_block_match.group(1) if code_block_match else text

    lines = candidate_text.strip().split("\n")
    all_tables: List[str] = []
    current_table: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]:
            current_table.append(stripped)
        else:
            if len(current_table) >= 2:
                all_tables.append("\n".join(current_table))
            current_table = []

    if len(current_table) >= 2:
        all_tables.append("\n".join(current_table))

    return all_tables


def extract_markdown_table_block(text: str) -> Optional[str]:
    """Extracts the markdown table section from text.
    If multiple tables exist, returns all tables separated by double newlines.
    """
    tables = extract_all_markdown_table_blocks(text)
    if not tables:
        return None
    return "\n\n".join(tables)


def _parse_single_table_to_dataframe(table_str: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """Parses a single markdown table block into a DataFrame."""
    try:
        lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return None, False

        # Header line
        header_line = lines[0]
        header_cells = [sanitize_cell_text(c) for c in header_line.split("|")[1:-1]]
        if not header_cells:
            return None, False

        # Separator line
        separator_line = lines[1]
        if not re.match(r"^\|?(\s*:?-+:?\s*\|?)+\s*$", separator_line):
            data_start_idx = 1
        else:
            data_start_idx = 2

        data_rows = []
        for line in lines[data_start_idx:]:
            row_cells = [sanitize_cell_text(c) for c in line.split("|")[1:-1]]
            if not row_cells:
                continue
            if len(row_cells) < len(header_cells):
                row_cells.extend([""] * (len(header_cells) - len(row_cells)))
            elif len(row_cells) > len(header_cells):
                row_cells = row_cells[: len(header_cells)]
            data_rows.append(row_cells)

        if not data_rows:
            return None, False

        df = pd.DataFrame(data_rows, columns=header_cells)

        # Attempt to convert numeric columns to float
        for col in df.columns:
            try:
                cleaned_col = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(cleaned_col)
            except Exception:
                pass

        return df, True
    except Exception:
        return None, False


def parse_all_markdown_tables(text: str) -> List[pd.DataFrame]:
    """Parses all markdown tables in text into a list of DataFrames."""
    blocks = extract_all_markdown_table_blocks(text)
    dfs = []
    for block in blocks:
        df, valid = _parse_single_table_to_dataframe(block)
        if valid and df is not None:
            dfs.append(df)
    return dfs


def parse_markdown_to_dataframe(text: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """Parses a markdown table string into a pandas DataFrame.
    If multiple tables exist, returns the concatenated DataFrame across tables.
    """
    dfs = parse_all_markdown_tables(text)
    if not dfs:
        return None, False
    if len(dfs) == 1:
        return dfs[0], True
    try:
        combined = pd.concat(dfs, ignore_index=True)
        return combined, True
    except Exception:
        return dfs[0], True


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame into a clean Markdown table."""
    return df.to_markdown(index=False)

