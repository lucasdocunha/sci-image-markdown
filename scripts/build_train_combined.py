"""Combine original Sci-ImageMiner figures with their augmented copies for training."""

import json
import os
from pathlib import Path

import click


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in ("id", "image", "table")
        ):
            raise click.ClickException(f"Invalid id/image/table in {path}:{line_number}")
        records.append(record)
    return records


@click.command()
@click.option("--processed-dir", default="data/processed", type=click.Path(exists=True, path_type=Path), show_default=True)
@click.option("--augmented-dir", default="data/augmented", type=click.Path(exists=True, path_type=Path), show_default=True)
@click.option("--output", default=None, type=click.Path(path_type=Path), show_default=True, help="Defaults to <augmented-dir>/train_combined.jsonl.")
@click.option("--overwrite", is_flag=True, default=False, help="Replace an existing combined file.")
def main(processed_dir: Path, augmented_dir: Path, output: Path | None, overwrite: bool) -> None:
    """Write train_combined.jsonl: augmented copies (relative paths) + originals (absolute paths)."""
    processed_dir, augmented_dir = processed_dir.resolve(), augmented_dir.resolve()
    output = output or (augmented_dir / "train_combined.jsonl")
    if output.exists() and not overwrite:
        raise click.ClickException(f"Output already exists (use --overwrite): {output}")

    copies = _read_jsonl(augmented_dir / "train.jsonl")
    originals = _read_jsonl(processed_dir / "train.jsonl")

    combined = []
    seen_ids: set[str] = set()
    # Copies first so the augmented distribution is immediately visible in the file.
    for record in copies:
        if record["id"] in seen_ids:
            raise click.ClickException(f"Duplicate id in sources: {record['id']}")
        seen_ids.add(record["id"])
        combined.append(record)
    for record in originals:
        if record["id"] in seen_ids:
            raise click.ClickException(f"Duplicate id in sources: {record['id']}")
        seen_ids.add(record["id"])
        # Absolute paths keep both origins loadable by the same dataset resolver.
        combined.append({**record, "image": str((processed_dir / record["image"]).resolve())})

    counts = {"copies": len(copies), "originals": len(originals)}
    click.echo(f"Combining {counts['copies']} augmented copies with {counts['originals']} originals.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for record in combined:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    click.echo(f"Wrote {len(combined)} records to {output}")


if __name__ == "__main__":
    main()
