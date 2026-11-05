"""
Structure-only targets: the table layout with every header and value hidden (TeleOCR-style).
"""

import re

STRUCTURE_INSTRUCTION = (
    "Write only the layout of the Markdown table(s) for the data plotted in this scientific figure: "
    "one row per data point and one column per variable or series, with x in every non-empty cell, "
    "including the header."
)

_SEPARATOR_CELL = re.compile(r"^:?-+:?$")


def _cells(line: str) -> list:
    inner = line.strip()[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def mask_table_structure(table: str) -> str:
    """Replaces every non-empty header/value cell with x, keeping separators, empty cells and blank lines."""
    lines = []
    for line in table.splitlines():
        if not line.strip().startswith("|"):
            lines.append(line)
            continue
        cells = _cells(line)
        if all(_SEPARATOR_CELL.match(cell) for cell in cells):
            lines.append(line.strip())
            continue
        lines.append("| " + " | ".join("x" if cell else "" for cell in cells) + " |")
    return "\n".join(lines)
