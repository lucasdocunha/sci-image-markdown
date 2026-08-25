"""
CLI tool for extracting markdown tables from scientific images.
"""

import os
import click
from rich.console import Console
from rich.markdown import Markdown

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.inference.predictor import TablePredictor

logger = setup_logger("predict")
console = Console()


@click.command()
@click.option("--image", required=True, help="Path to input image file.")
@click.option("--config", default="configs/default.yaml", help="Path to base configuration YAML.")
@click.option("--adapter-path", default=None, help="Optional path to trained PEFT LoRA adapter.")
@click.option("--output-csv", default=None, help="Optional path to save extracted table as CSV.")
@click.option("--output-md", default=None, help="Optional path to save extracted table as Markdown.")
def main(image: str, config: str, adapter_path: str, output_csv: str, output_md: str):
    """Runs inference on a single scientific figure image."""
    if not os.path.exists(image):
        raise FileNotFoundError(f"Input image not found: {image}")

    cfg = load_config(config)
    logger.info(f"Loading predictor for image: {image}")
    predictor = TablePredictor(cfg, adapter_path=adapter_path)

    result = predictor.predict_image(image)
    markdown_table = result["markdown_table"]

    console.print("\n[bold green]Extracted Markdown Table:[/bold green]\n")
    console.print(Markdown(markdown_table))

    if output_md:
        os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(markdown_table)
        logger.info(f"Saved markdown to: {output_md}")

    if output_csv and result["dataframe"] is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        result["dataframe"].to_csv(output_csv, index=False)
        logger.info(f"Saved CSV to: {output_csv}")


if __name__ == "__main__":
    main()
