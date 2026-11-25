"""
Decomposição de tabelas Markdown em três subtarefas (experimento B):

- M1 (skeleton): estrutura da tabela — cabeçalhos e número de linhas/colunas, com
  as células de dados substituídas por PLACEHOLDER.
- M2 (values):   apenas os valores das linhas de dados, sem cabeçalho.
- M3 (fusion):   texto puro — recebe esqueleto + valores e devolve a tabela final.

As funções aqui são puras (sem modelo), para que o round-trip
fill_skeleton(make_skeleton(gt), make_values(gt)) == gt possa ser testado.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.io import read_jsonl

PLACEHOLDER = "?"

SKELETON_SYSTEM_PROMPT = (
    "You extract only the STRUCTURE of the data table in the given scientific figure as a Markdown table. "
    "Keep the exact column headers and the exact number of rows and columns. "
    "Replace every data cell (not the headers) with '?'. Output the Markdown table only."
)

VALUES_SYSTEM_PROMPT = (
    "You extract only the DATA VALUES plotted in the given scientific figure. Output them as rows of a "
    "Markdown table WITHOUT a header row, in the same order as the table body (row by row, left to right). "
    "Output clean numbers only, no approximation symbols. Output the Markdown rows only."
)

FUSION_SYSTEM_PROMPT = (
    "You merge a Markdown table skeleton with extracted values. Replace every '?' cell in the skeleton "
    "with the matching value, in order, and output the complete Markdown table(s) only."
)


# --------------------------------------------------------------------------- #
# Parsing / rendering helpers
# --------------------------------------------------------------------------- #

def split_blocks(md: str) -> List[List[str]]:
    """Separa um Markdown com uma ou mais tabelas (separadas por linha em branco) em listas de linhas."""
    blocks: List[List[str]] = []
    for chunk in re.split(r"\n\s*\n", (md or "").strip()):
        lines = [ln.strip() for ln in chunk.strip().split("\n") if ln.strip()]
        if lines:
            blocks.append(lines)
    return blocks


def split_cells(line: str) -> List[str]:
    """'| a | b |' -> ['a', 'b']."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|")


def render_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table_head_and_body(lines: List[str]) -> Tuple[List[str], List[str]]:
    """Retorna (linhas de cabeçalho incluindo separador, linhas de corpo)."""
    if len(lines) >= 2 and is_separator(lines[1]):
        return lines[:2], lines[2:]
    return lines[:1], lines[1:]


def _strip_separators(lines: List[str]) -> List[str]:
    return [ln for ln in lines if not is_separator(ln)]


# --------------------------------------------------------------------------- #
# Conversões ground-truth -> subtarefas
# --------------------------------------------------------------------------- #

def make_skeleton(md: str) -> str:
    """Esqueleto: cabeçalhos preservados, células de dados viram PLACEHOLDER."""
    out: List[str] = []
    for lines in split_blocks(md):
        if not is_table_line(lines[0]):
            continue
        head, body = _table_head_and_body(lines)
        new_body = []
        for row in body:
            n = len(split_cells(row))
            new_body.append(render_row([PLACEHOLDER] * n))
        out.append("\n".join(head + new_body))
    return "\n\n".join(out)


def make_values(md: str) -> str:
    """Valores: apenas as linhas de corpo de cada tabela, sem cabeçalho."""
    out: List[str] = []
    for lines in split_blocks(md):
        if not is_table_line(lines[0]):
            continue
        _, body = _table_head_and_body(lines)
        out.append("\n".join(render_row(split_cells(row)) for row in body))
    return "\n\n".join(out)


# --------------------------------------------------------------------------- #
# Fusão determinística (referência / fallback do M3)
# --------------------------------------------------------------------------- #

def parse_values(values_md: str) -> List[List[List[str]]]:
    """Saída do M2 -> lista (por tabela) de linhas de células. Ignora separadores e texto solto."""
    tables: List[List[List[str]]] = []
    for lines in split_blocks(values_md):
        rows = [split_cells(ln) for ln in _strip_separators(lines) if is_table_line(ln)]
        if rows:
            tables.append(rows)
    return tables


def fill_skeleton(skeleton_md: str, values_md: str) -> str:
    """Preenche cada PLACEHOLDER do esqueleto com o valor posicional (tabela, linha, coluna).

    Tolerante a divergências: se faltar valor, a célula fica vazia; valores sobrando são ignorados.
    """
    value_tables = parse_values(values_md)
    out: List[str] = []
    for t_idx, lines in enumerate(split_blocks(skeleton_md)):
        if not is_table_line(lines[0]):
            continue
        head, body = _table_head_and_body(lines)
        v_rows = value_tables[t_idx] if t_idx < len(value_tables) else []
        new_body = []
        for r_idx, row in enumerate(body):
            cells = split_cells(row)
            v_cells = v_rows[r_idx] if r_idx < len(v_rows) else []
            filled = []
            for c_idx, cell in enumerate(cells):
                if cell == PLACEHOLDER:
                    filled.append(v_cells[c_idx] if c_idx < len(v_cells) else "")
                else:
                    filled.append(cell)
            new_body.append(render_row(filled))
        out.append("\n".join(head + new_body))
    return "\n\n".join(out)


def table_shapes(md: str) -> List[Tuple[int, int]]:
    """(linhas de dados, colunas) de cada tabela do Markdown — métrica intermediária do M1."""
    shapes: List[Tuple[int, int]] = []
    for lines in split_blocks(md):
        if not is_table_line(lines[0]):
            continue
        _, body = _table_head_and_body(lines)
        shapes.append((len(body), len(split_cells(lines[0]))))
    return shapes


def body_cells(md: str) -> List[List[List[str]]]:
    """Células do corpo de cada tabela (sem cabeçalho) — referência para a métrica do M2."""
    out: List[List[List[str]]] = []
    for lines in split_blocks(md):
        if not is_table_line(lines[0]):
            continue
        _, body = _table_head_and_body(lines)
        out.append([split_cells(row) for row in body])
    return out


# --------------------------------------------------------------------------- #
# M3: prompt de fusão e dataset só-texto
# --------------------------------------------------------------------------- #

def build_fusion_user_text(skeleton_md: str, values_md: str) -> str:
    return (
        "Skeleton of the table ('?' cells must be filled):\n"
        f"{skeleton_md.strip()}\n\n"
        "Extracted values (data rows, in the same order as the '?' cells):\n"
        f"{values_md.strip()}\n\n"
        "Output the complete final Markdown table."
    )


def clean_model_output(text: str) -> str:
    """Remove cercas de código e espaços extras da saída crua de um modelo."""
    lines = [ln.rstrip() for ln in (text or "").strip().split("\n")]
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


class FusionTextDataset:
    """Dataset só-texto para o M3.

    Cada registro JSONL precisa de: 'skeleton', 'values' e 'table' (alvo = GT final).
    Retorna o mesmo formato de item que SciImageTableDataset, com image=None, para que
    o LazyMultimodalDataset do trainer monte a conversa só com texto.
    """

    def __init__(self, data_path: str, system_prompt: Optional[str] = None, **_: Any):
        self.records = read_jsonl(data_path)
        self.system_prompt = system_prompt or FUSION_SYSTEM_PROMPT

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        return {
            "id": rec.get("id", str(idx)),
            "image": None,
            "table": rec["table"],
            "system_prompt": self.system_prompt,
            "user_text": build_fusion_user_text(rec["skeleton"], rec["values"]),
            "metadata": {},
        }
