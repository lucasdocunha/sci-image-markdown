"""Append structure-only copies of the real train figures to a training jsonl."""

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.structure import STRUCTURE_INSTRUCTION, mask_table_structure  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@click.command()
@click.option("--base", default="data/synthetic_723/train_combined.jsonl", type=click.Path(exists=True, path_type=Path), show_default=True, help="Training file the copies are appended to.")
@click.option("--real", default="data/processed/train.jsonl", type=click.Path(exists=True, path_type=Path), show_default=True, help="Real figures to mask.")
@click.option("--output", default="data/synthetic_723/train_structure.jsonl", type=click.Path(path_type=Path), show_default=True)
@click.option("--overwrite", is_flag=True, default=False, help="Replace an existing output file.")
def main(base: Path, real: Path, output: Path, overwrite: bool) -> None:
    """Write <base> + one masked copy (headers and values hidden, own instruction) per real figure."""
    if output.exists() and not overwrite:
        raise click.ClickException(f"Output already exists (use --overwrite): {output}")

    records = _read_jsonl(base)
    seen_ids = {record["id"] for record in records}
    copies = []
    for record in _read_jsonl(real):
        copy_id = f"{record['id']}__structure"
        if copy_id in seen_ids:
            raise click.ClickException(f"Duplicate id: {copy_id}")
        copies.append({
            "id": copy_id,
            # Absolute path: the base file resolves images relative to its own folder.
            "image": str((real.parent / record["image"]).resolve()),
            "table": mask_table_structure(record["table"]),
            "instruction": STRUCTURE_INSTRUCTION,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records + copies:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    click.echo(f"Wrote {len(records)} base + {len(copies)} structure-only records to {output}")


if __name__ == "__main__":
    main()
