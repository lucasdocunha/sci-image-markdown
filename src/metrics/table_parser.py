"""
Robust Markdown table parsing and DataFrame extraction.
"""

import io
import re
from typing import Optional, Tuple, List
import pandas as pd


def extract_markdown_table_block(text: str) -> Optional[str]:
    """Extracts the markdown table section from generated model text.
    
    Handles markdown code blocks (```markdown ... ``` or ``` ... ```) and
    raw pipe-delimited table patterns.
    """
    if not text or not isinstance(text, str):
        return None

    # Check for markdown code fences first
    code_block_match = re.search(r"```(?:markdown|table)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate_text = code_block_match.group(1) if code_block_match else text

    # Look for pipe-separated table lines (| ... | ... |)
    lines = candidate_text.strip().split("\n")
    table_lines: List[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]:
            in_table = True
            table_lines.append(stripped)
        elif in_table:
            # End of table encountered
            break

    if len(table_lines) >= 2:
        return "\n".join(table_lines)

    return None


def parse_markdown_to_dataframe(text: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """Parses a markdown table string into a pandas DataFrame.
    
    Returns:
        Tuple of (DataFrame or None, is_valid_boolean)
    """
    table_str = extract_markdown_table_block(text)
    if not table_str:
        return None, False

    try:
        lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]
        if len(lines) < 2:
            return None, False

        # Header line
        header_line = lines[0]
        header_cells = [c.strip() for c in header_line.split("|")[1:-1]]
        if not header_cells:
            return None, False

        # Check separator line (e.g. |---|---| or |:---|---:|)
        separator_line = lines[1]
        if not re.match(r"^\|?(\s*:?-+:?\s*\|?)+\s*$", separator_line):
            # Not a standard separator, but let's check if rows start immediately
            data_start_idx = 1
        else:
            data_start_idx = 2

        data_rows = []
        for line in lines[data_start_idx:]:
            row_cells = [c.strip() for c in line.split("|")[1:-1]]
            if not row_cells:
                continue
            # Pad or truncate to match header length
            if len(row_cells) < len(header_cells):
                row_cells.extend([""] * (len(header_cells) - len(row_cells)))
            elif len(row_cells) > len(header_cells):
                row_cells = row_cells[:len(header_cells)]
            data_rows.append(row_cells)

        if not data_rows:
            return None, False

        df = pd.DataFrame(data_rows, columns=header_cells)

        # Attempt to convert numeric columns to float
        for col in df.columns:
            try:
                # Remove common formatting like commas or units if applicable
                cleaned_col = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(cleaned_col)
            except Exception:
                pass

        return df, True
    except Exception:
        return None, False


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame into a clean Markdown table."""
    return df.to_markdown(index=False)
