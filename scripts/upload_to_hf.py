"""
Upload script to publish the fine-tuned Qwen2.5-VL-3B LoRA adapter and extraction codebase to Hugging Face Hub.
"""

import os
import shutil
import click
from huggingface_hub import HfApi, get_token


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
pipeline_tag: image-to-text
tags:
- vision-language
- image-to-text
- table-extraction
- scientific-figures
- markdown-table
- qwen2.5-vl
- lora
- peft
- icdar-metric-loss
language:
- en
metrics:
- teds
- cell_precision
- cell_f1
- valid_table
---

# Sci-Image-Markdown: Qwen2.5-VL-3B LoRA Table Extractor

Fine-tuned PEFT LoRA adapter for **Qwen2.5-VL-3B-Instruct** designed to extract quantitative plotted data from scientific figure panels directly into clean, structured Markdown tables.

This model was trained using an **ICDAR Metric-Aware Loss** (combining Differentiable Token-Weighted Numerical & Structural loss with fast in-VRAM GPU rewards for Table Edit Distance and Cell RMS Error).

## 📊 Benchmark Results (Test Set - 373 Scientific Figures)

| Metric | Base Model (Zero-Shot) | Standard SFT (Cross-Entropy) | **Our LoRA (ICDAR Metric Loss)** |
| :--- | :---: | :---: | :---: |
| **Valid Markdown Table** | 88.74% | 99.73% | **99.73%** |
| **Cell Precision** | 21.08% | 21.37% | **39.58%** *(+85.2%)* |
| **Cell F1-Score** | 18.42% | 20.60% | **34.67%** *(+68.3%)* |
| **TEDS (Tree Edit Distance)** | 23.16% | 23.16% | **36.24%** *(+56.5%)* |
| **ICDAR Numerical RMS** | 48.12% | 48.12% | **51.66%** *(+7.4%)* |
| **ICDAR Composite Score** | 35.64% | 35.64% | **43.95%** *(+23.3%)* |

---

## 🚀 Quickstart & Usage

### Installation

```bash
pip install torch transformers peft qwen-vl-utils pillow torchvision bitsandbytes
```

### Inference Example

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_id = "{repo_id}"

# 1. Load processor
processor = AutoProcessor.from_pretrained(adapter_id)

# 2. Load 4-bit base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Load trained LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

# 4. Extract table from figure
image_path = "figure.png"
image = Image.open(image_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract the plotted quantitative data into a clean Markdown table with column headers."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

print("--- Extracted Markdown Table ---")
print(output_text)
```

---

## 🛠 Training Details

- **Base Architecture:** `Qwen2.5-VL-3B-Instruct`
- **Method:** 4-bit QLoRA ($r=16, \\alpha=32$)
- **Hardware:** Single NVIDIA GeForce GTX 1660 (6 GB VRAM)
- **Loss:** Token-Weighted ICDAR Loss (Proxy for Tree Edit Distance and Relative Numerical Error)
- **Precision:** 4-bit NF4 base weights, FP16 LoRA compute

## 📜 Citation & License

- Base Model License: [Qwen Research License / Apache 2.0](https://github.com/QwenLM/Qwen2.5-VL)
- Codebase: [sci-image-markdown](https://github.com/lucasdocunha/sci-image-markdown)
"""


INFERENCE_EXAMPLE_SCRIPT = """# Standalone inference example using the Hugging Face adapter
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import sys

def main(image_path: str, adapter_id: str = "."):
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"Loading processor for {adapter_id}...")
    processor = AutoProcessor.from_pretrained(adapter_id)

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading PEFT LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract the plotted quantitative data into a clean Markdown table with column headers."}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    print("\\n=== Extracted Markdown Table ===\\n")
    print(output_text)

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "figure.png"
    main(img)
"""


@click.command()
@click.option("--repo-id", default=None, help="Hugging Face repository ID (e.g., 'username/sci-image-markdown-qwen2.5-vl-3b').")
@click.option("--token", default=None, help="Hugging Face write token (or set HF_TOKEN env var).")
@click.option("--adapter-dir", default="outputs/checkpoints_icdar_loss/final_adapter", help="Path to local trained adapter.")
@click.option("--export-dir", default="outputs/hf_export", help="Temporary staging directory for packaging.")
@click.option("--private", is_flag=True, default=False, help="Create a private repository on Hugging Face.")
@click.option("--include-code", is_flag=True, default=True, help="Include project source code and configs in the upload.")
def main(
    repo_id: str,
    token: str,
    adapter_dir: str,
    export_dir: str,
    private: bool,
    include_code: bool,
):
    """Packages and uploads the trained adapter and extraction codebase to Hugging Face Hub."""
    hf_token = token or os.getenv("HF_TOKEN") or get_token()
    if not hf_token:
        click.echo("[ERROR] No Hugging Face token found.")
        click.echo("Please pass --token YOUR_TOKEN, run 'huggingface-cli login', or export HF_TOKEN=YOUR_TOKEN.")
        return

    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info.get("name", user_info.get("fullname", "user"))
    click.echo(f"Authenticated as Hugging Face user: {username}")

    if not repo_id:
        repo_id = f"{username}/sci-image-markdown-qwen2.5-vl-3b"
        click.echo(f"No --repo-id specified. Using default: {repo_id}")

    if not os.path.exists(adapter_dir):
        click.echo(f"[ERROR] Adapter directory not found: {adapter_dir}")
        return

    # Prepare export directory
    click.echo(f"Preparing package in {export_dir}...")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # 1. Copy adapter files
    for item in os.listdir(adapter_dir):
        s = os.path.join(adapter_dir, item)
        d = os.path.join(export_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # 2. Write curated README.md Model Card
    card_content = MODEL_CARD_TEMPLATE.replace("{repo_id}", repo_id)
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card_content)

    # 3. Add standalone inference example script
    with open(os.path.join(export_dir, "inference_example.py"), "w", encoding="utf-8") as f:
        f.write(INFERENCE_EXAMPLE_SCRIPT)

    # 4. Optionally bundle src and configs
    if include_code:
        click.echo("Bundling src/ and configs/ into repository...")
        if os.path.exists("src"):
            shutil.copytree("src", os.path.join(export_dir, "src"), dirs_exist_ok=True)
        if os.path.exists("configs"):
            shutil.copytree("configs", os.path.join(export_dir, "configs"), dirs_exist_ok=True)
        if os.path.exists("requirements.txt"):
            shutil.copy2("requirements.txt", os.path.join(export_dir, "requirements.txt"))

    # 5. Create repository on Hugging Face
    click.echo(f"Creating repository on Hugging Face: {repo_id} (private={private})...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    # 6. Upload folder
    click.echo(f"Uploading files to https://huggingface.co/{repo_id}...")
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: initial release of sci-image-markdown Qwen2.5-VL-3B LoRA adapter and extraction codebase",
    )

    click.echo("\n" + "=" * 80)
    click.echo(f"  SUCCESSFULLY UPLOADED TO HUGGING FACE!")
    click.echo(f"  Model URL: https://huggingface.co/{repo_id}")
    click.echo("=" * 80 + "\n")


if __name__ == "__main__":
    main()
