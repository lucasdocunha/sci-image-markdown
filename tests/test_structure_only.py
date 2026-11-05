"""Tests for structure-only training samples (masked tables with their own instruction)."""

from PIL import Image

from src.data.preprocessor import format_qwen_vl_conversation
from src.data.structure import STRUCTURE_INSTRUCTION, mask_table_structure
from src.training.trainer import LazyMultimodalDataset


TWO_TABLES = (
    "| Temperature (°C) | GPC (Å/cycle) | Note |\n"
    "|---|:---:|---|\n"
    "| 100 | 1.2 |  |\n"
    "| 200 | 1.5 | peak |\n"
    "\n"
    "| Cycles | Thickness (nm) |\n"
    "|------|-----|\n"
    "| 10 | 1.1 |"
)


def test_mask_hides_headers_and_values_but_keeps_the_layout():
    masked = mask_table_structure(TWO_TABLES)

    assert masked == (
        "| x | x | x |\n"
        "|---|:---:|---|\n"
        "| x | x |  |\n"
        "| x | x | x |\n"
        "\n"
        "| x | x |\n"
        "|------|-----|\n"
        "| x | x |"
    )


def test_conversation_uses_the_given_instruction_and_keeps_the_default_otherwise():
    image = Image.new("RGB", (8, 8))

    default = format_qwen_vl_conversation(image, "| a |")
    structure = format_qwen_vl_conversation(image, "| x |", instruction=STRUCTURE_INSTRUCTION)

    assert default[1]["content"][1]["text"].startswith("Extract all numerical data points")
    assert structure[1]["content"][1]["text"] == STRUCTURE_INSTRUCTION


def test_training_dataset_passes_the_record_instruction_through():
    image = Image.new("RGB", (8, 8))
    records = [
        {"image": image, "table": "| a |", "system_prompt": "sys"},
        {"image": image, "table": "| x |", "system_prompt": "sys", "instruction": STRUCTURE_INSTRUCTION},
    ]

    dataset = LazyMultimodalDataset(records)

    assert dataset[0]["messages"][1]["content"][1]["text"].startswith("Extract all numerical data points")
    assert dataset[1]["messages"][1]["content"][1]["text"] == STRUCTURE_INSTRUCTION
