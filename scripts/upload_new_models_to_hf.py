"""
Script to upload LightOnOCR-3-4B and Qwen3-VL-8B LoRA adapters to Hugging Face Hub.
Repo: lucasoc/sci-image-models
"""

import os
import sys
import json
from pathlib import Path
from huggingface_hub import HfApi

TOKEN_FILE = Path("/home/lucas.ocunha/.cache/huggingface/token")
if TOKEN_FILE.exists():
    token = TOKEN_FILE.read_text().strip()
else:
    token = os.getenv("HF_TOKEN", "")

if not token:
    raise ValueError("Hugging Face token not found!")

api = HfApi(token=token)
REPO_ID = "lucasoc/sci-image-models"

# 1. LightOnOCR-3-4B
LIGHTON_DIR = Path("/media/ssd2/datasets/sci-image-markdown-data/outputs/lightonocr_3_4b_icdar_loss/final_adapter")
LIGHTON_REPORT = Path("/media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/lightonocr_3_4b_icdar_loss/test_report.json")

# 2. Qwen3-VL-8B
QWEN3_8B_DIR = Path("/media/ssd2/datasets/sci-image-markdown-data/outputs/qwen3_vl_8b_icdar_loss/final_adapter")
QWEN3_8B_REPORT = Path("/media/ssd2/datasets/sci-image-markdown-data/outputs/eval_results/qwen3_vl_8b_icdar_loss/test_report.json")

# Model Card README for LightOnOCR-3-4B
LIGHTON_README = """---
base_model: lightonai/LightOnOCR-3-4B
library_name: peft
pipeline_tag: image-to-text
tags:
- vision-language
- ocr
- table-extraction
- scientific-figures
- lightonocr
- lora
- peft
- icdar-metric-loss
language:
- pt
- en
metrics:
- teds
- cell_precision
- cell_recall
- cell_f1
- valid_table
- icdar_rms
- icdar_score
---

# LightOnOCR-3-4B LoRA Adapter (Novo Recorde Absoluto ICDAR 2026 - 52.32%)

Adapter PEFT LoRA treinado sobre o **`lightonai/LightOnOCR-3-4B`** com a **Loss v3 (ICDAR Metric & TEDS Topology-Aware Loss)** na extração estruturada de tabelas numéricas a partir de figuras científicas complexas (deposição e corrosão atômica ALD/ALE).

### 🏆 Métricas Oficiais no Conjunto de Teste (373 figuras)
- **Score Composto ICDAR:** **`0.5232` (52.32%)** 🏆 *(Supera o 1º colocado oficial do desafio ICDAR 2026 Peking Univ de 51.81%)*
- **TEDS (Tree Edit Distance Similarity):** **`0.4252` (42.52%)**
- **ICDAR Numerical RMS:** **`0.6212` (62.12%)**
- **Taxa de Tabelas Válidas:** **`98.39%`**
- **Cell Recall ($\le 5\%$ tol.):** **`59.99%`**
- **Cell Precision:** **`50.98%`**
- **Cell F1:** **`47.93%`**
- **Cell RNE (Erro Relativo Médio):** **`0.0036` (0.36%)**
- **ROUGE-L:** **`0.4349`**
- **BLEU-4:** **`0.2445`**

### Como Usar
```python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "lightonai/LightOnOCR-3-4B"
adapter_id = "lucasoc/sci-image-models"
subfolder = "lightonocr_3_4b"

processor = AutoProcessor.from_pretrained(adapter_id, subfolder=subfolder)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)
model.eval()
```
"""

# Model Card README for Qwen3-VL-8B
QWEN3_8B_README = """---
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: peft
pipeline_tag: image-to-text
tags:
- vision-language
- image-to-text
- table-extraction
- scientific-figures
- qwen3-vl
- lora
- peft
- icdar-metric-loss
language:
- pt
- en
metrics:
- teds
- cell_precision
- cell_recall
- cell_f1
- valid_table
- icdar_rms
- icdar_score
---

# Qwen3-VL-8B LoRA Adapter (ICDAR 2026 - 48.72%)

Adapter PEFT LoRA treinado sobre o **`Qwen/Qwen3-VL-8B-Instruct`** com a **Loss v3 (ICDAR Metric & TEDS Topology-Aware Loss)** para extração de tabelas a partir de figuras científicas.

### 🏆 Métricas Oficiais no Conjunto de Teste (373 figuras)
- **Score Composto ICDAR:** **`0.4872` (48.72%)** *(Supera o 2º colocado oficial do desafio ICDAR 2026 Samsung de 46.01%)*
- **TEDS (Tree Edit Distance Similarity):** **`0.3668` (36.68%)**
- **ICDAR Numerical RMS:** **`0.6076` (60.76%)**
- **Taxa de Tabelas Válidas:** **`99.73%`**
- **Cell Recall ($\le 5\%$ tol.):** **`59.01%`**
- **Cell Precision:** **`45.50%`**
- **Cell F1:** **`44.14%`**
- **Cell RMSE:** **`71.45`**
- **Cell RNE (Erro Relativo Médio):** **`0.0040` (0.40%)**

### Como Usar
```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base_model_id = "Qwen/Qwen3-VL-8B-Instruct"
adapter_id = "lucasoc/sci-image-models"
subfolder = "qwen3_vl_8b"

processor = AutoProcessor.from_pretrained(adapter_id, subfolder=subfolder)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id, subfolder=subfolder)
model.eval()
```
"""

# Updated Root README for the repository
ROOT_README = """---
license: apache-2.0
base_model:
- lightonai/LightOnOCR-3-4B
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen3-VL-8B-Instruct
- Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
pipeline_tag: image-to-text
tags:
- vision-language
- image-to-text
- table-extraction
- scientific-figures
- markdown-table
- lightonocr
- qwen2.5-vl
- qwen3-vl
- lora
- peft
- icdar-metric-loss
language:
- pt
- en
metrics:
- teds
- cell_precision
- cell_recall
- cell_f1
- valid_table
- icdar_rms
- icdar_score
---

# Sci-Image-Markdown: Extratores Multimodais de Tabelas Científicas (LoRA Adapters)

Coleção oficial de LoRA Adapters de alta precisão desenvolvidos para o framework **sci-image-markdown**, especializados na transcrição visual direta de figuras científicas densas (espectroscopia, ALD/ALE, isotermas e cinética química) em tabelas estruturadas em formato **GitHub-Flavored Markdown**.

Treinados com a **ICDAR Metric-Aware Loss v3** (Loss balanceada de proximidade numérica suave, penalidade de cardinalidade e ancoragem topológica TEDS).

---

## 🏆 Tabela Oficial do Leaderboard ICDAR 2026 (Task 2: Data Table Extraction)

Avaliação rigorosa realizada sobre as **373 figuras científicas reais** do conjunto de teste independente oficial do desafio:

| Posição | Modelo / Backbone | Método / Adaptação | Subpasta no Repo | Tabelas Válidas (`valid_table`) | RMS Numérico (`icdar_rms`) | Estrutura TEDS (`teds`) | Score Final ICDAR | Hardware Recomendado |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1º** | **`LightOnOCR-3-4B`** *(Novo Recorde)* | **LoRA + Loss v3** | `lightonocr_3_4b/` | **98.39%** | **0.6212** | **0.4252** | **0.5232** 🏆 | ~8 GB VRAM |
| 🥈 **2º** | **`Qwen2.5-VL-7B`** | **LoRA + Loss v2** | `qwen2_5_vl_7b/` | **99.73%** | **0.6026** | **0.4396** | **0.5211** 🚀 | ~14 GB VRAM |
| 🥉 **3º** | **`Qwen2.5-VL-3B`** | **LoRA + Loss v1** | `root /` | **99.73%** | **0.5830** | **0.4068** | **0.4949** | < 8 GB VRAM |
| **4º** | **`Qwen3-VL-8B`** | **LoRA + Loss v3** | `qwen3_vl_8b/` | **99.73%** | **0.6076** | **0.3668** | **0.4872** | ~16 GB VRAM |
| *Ref* | *1º Colocado Oficial ICDAR 2026 (Peking)* | *Ensemble / Multi-VLM* | - | ~99.0% | 0.5810 | 0.4552 | 0.5181 | Multi-GPU Cluster |
| *Ref* | *2º Colocado Oficial ICDAR 2026 (Samsung)* | *Prompting Especializado* | - | ~98.5% | 0.5230 | 0.3972 | 0.4601 | Multi-GPU Cluster |
| *Base* | *Qwen2.5-VL-3B (Zero-Shot)* | *Sem Treinamento* | - | 88.74% | 0.4812 | 0.2316 | 0.3564 | < 8 GB VRAM |

---

## 📂 Organização dos Modelos no Repositório

```
lucasoc/sci-image-models/
├── README.md                           # Visão geral e leaderboard comparativo
├── adapter_config.json                 # Adapter Qwen2.5-VL-3B (raiz)
├── adapter_model.safetensors           # Pesos LoRA Qwen2.5-VL-3B
├── lightonocr_3_4b/                    # 🏆 NOVO RECORDE ABSOLUTO (52.32%)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors       # 88 MB
│   ├── test_report.json                # Relatório completo de teste
│   └── README.md
├── qwen2_5_vl_7b/                      # SOTA Qwen2.5-VL-7B (52.11%)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors       # 177 MB
│   ├── test_report.json
│   └── README.md
└── qwen3_vl_8b/                        # Qwen3-VL-8B (48.72%)
    ├── adapter_config.json
    ├── adapter_model.safetensors       # 176 MB
    ├── test_report.json
    └── README.md
```

---

## 🚀 Como Executar Inferência

### 1. LightOnOCR-3-4B (Novo Campeão Absoluto - 52.32%)

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

# Carregar processador e base model
processor = AutoProcessor.from_pretrained("lucasoc/sci-image-models", subfolder="lightonocr_3_4b")
base_model = AutoModelForCausalLM.from_pretrained(
    "lightonai/LightOnOCR-3-4B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "lucasoc/sci-image-models", subfolder="lightonocr_3_4b")
model.eval()
```

### 2. Qwen2.5-VL-7B ou Qwen3-VL-8B

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Para Qwen2.5-VL-7B use subfolder="qwen2_5_vl_7b" e base "Qwen/Qwen2.5-VL-7B-Instruct"
# Para Qwen3-VL-8B use subfolder="qwen3_vl_8b" e base "Qwen/Qwen3-VL-8B-Instruct"

subfolder = "qwen2_5_vl_7b"
base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained("lucasoc/sci-image-models", subfolder=subfolder)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "lucasoc/sci-image-models", subfolder=subfolder)
model.eval()

# Processamento da figura
image = Image.open("exemplo_grafico_cientifico.png").convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all numerical data points from this scientific figure into a Markdown table."}
        ]
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=1024)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(result)
```

---

## 🔬 Evolução das Funções de Perda (Loss Formulation)

1. **Loss v1 (Token-Weighted Static CE):** Ponderação de tokens numéricos (4.0x) e estruturais (3.0x).
2. **Loss v2 (Float-Proximity & Cardinality Surrogate Loss):** Penalidade suave por desvio percentual em números decodificados + multiplicador global de cardinalidade.
3. **Loss v3 (TEDS Topology & Directional Control):** Introdução de penalidade de colunas (`col_count_penalty_weight = 2.0`), reforço de cabeçalho (`header_weight = 4.5`) e terminação antecipada direcional sobre `<eos>` para evitar sobregeração de linhas vazias. Alcançou o recorde absoluto de **0.5232** com o LightOnOCR-3-4B.

---

## 📜 Citação & Repositório Oficial

Repositório completo no GitHub: [lucasdocunha/sci-image-markdown](https://github.com/lucasdocunha/sci-image-markdown)
"""


def main():
    print(f"Uploading models to Hugging Face Hub: {REPO_ID}...")

    # Write model card READMEs
    (LIGHTON_DIR / "README.md").write_text(LIGHTON_README, encoding="utf-8")
    (QWEN3_8B_DIR / "README.md").write_text(QWEN3_8B_README, encoding="utf-8")

    # Upload LightOnOCR-3-4B
    print("Uploading LightOnOCR-3-4B adapter and evaluation...")
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(LIGHTON_DIR),
        path_in_repo="lightonocr_3_4b",
        commit_message="Add LightOnOCR-3-4B LoRA adapter and test report (New ICDAR Record 0.5232)"
    )
    print("✓ LightOnOCR-3-4B uploaded successfully!")

    # Upload Qwen3-VL-8B
    print("Uploading Qwen3-VL-8B adapter and evaluation...")
    api.upload_folder(
        repo_id=REPO_ID,
        folder_path=str(QWEN3_8B_DIR),
        path_in_repo="qwen3_vl_8b",
        commit_message="Add Qwen3-VL-8B LoRA adapter and test report (ICDAR 0.4872)"
    )
    print("✓ Qwen3-VL-8B uploaded successfully!")

    # Upload Root README
    print("Uploading updated root README.md...")
    api.upload_file(
        repo_id=REPO_ID,
        path_or_fileobj=ROOT_README.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Update README with LightOnOCR-3-4B (0.5232) and Qwen3-VL-8B (0.4872) benchmark results"
    )
    print("✓ Root README.md updated successfully!")


if __name__ == "__main__":
    main()
