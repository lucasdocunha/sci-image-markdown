"""
I/O utilities for reading and writing dataset files (JSONL, JSON, CSV).
"""

import json
import os
from typing import Any, Dict, List, Generator


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Reads a JSONL file into a list of dictionaries."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Yields records one by one from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    """Writes a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
